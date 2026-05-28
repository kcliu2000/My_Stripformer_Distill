import wandb
import numpy as np
import logging
from functools import partial
import os
import cv2
import torch
import torch.optim as optim
import tqdm
import yaml
from joblib import cpu_count
from torch.utils.data import DataLoader
import random
from dataset import PairedDataset
from metric_counter import MetricCounter
from models.losses import get_loss
from models.models import get_model
from models.networks import get_nets
from torch.optim.lr_scheduler import CosineAnnealingLR
cv2.setNumThreads(0)


class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.metric_counter = MetricCounter(config['experiment_desc'])

    def train(self):
        self._init_params()
        start_epoch = 0
        if os.path.exists('last_Stripformer_gopro.pth'):
            print('load_pretrained')
            training_state = torch.load('last_Stripformer_gopro.pth')
            start_epoch = training_state['epoch']
            
            new_weight = self.netG.state_dict()
            new_weight.update(training_state['model_state'])
            self.netG.load_state_dict(new_weight)
            
            new_optimizer = self.optimizer_G.state_dict()
            new_optimizer.update(training_state['optimizer_state'])
            self.optimizer_G.load_state_dict(new_optimizer)
            
            new_scheduler = self.scheduler_G.state_dict()
            new_scheduler.update(training_state['scheduler_state'])
            self.scheduler_G.load_state_dict(new_scheduler)

        for epoch in range(start_epoch, self.config['num_epochs']):
            self._run_epoch(epoch)
            self.validate()
            # 儲存 Checkpoint
            torch.save({'epoch': epoch + 1,
                        'model_state': self.netG.state_dict(),
                        'optimizer_state': self.optimizer_G.state_dict(),
                        'scheduler_state': self.scheduler_G.state_dict()},
                       'last_Stripformer_gopro.pth')

    def _init_params(self):
        # ========================================
        # 1. 學生模型 (Student) 建立與初始化
        # ========================================
        self.criterionG = get_loss(self.config['model'])
        self.netG = get_nets(self.config['model'])
        self.netG.cuda()

        # ========================================
        # 2. 老師模型 (Teacher) 建立與載入
        # ========================================
        # 讀取 yaml 中的 teacher_model 設定，若無則預設與學生相同(防呆)
        teacher_config = self.config.get('teacher_model', self.config['model'])
        self.teacher_model = get_nets(teacher_config)
        self.teacher_model.cuda()

        teacher_weight_path = './pretrained_models/Stripformer_gopro.pth'
        if os.path.exists(teacher_weight_path):
            print(f"✅ Loading Teacher Weights from {teacher_weight_path}")
            teacher_state = torch.load(teacher_weight_path)
            # 相容不同存檔格式
            if 'model_state' in teacher_state:
                self.teacher_model.load_state_dict(teacher_state['model_state'])
            else:
                self.teacher_model.load_state_dict(teacher_state)
        else:
            print(f"⚠️ 找不到老師權重: {teacher_weight_path} (這將導致蒸餾失效！)")

        # 老師進入 eval 模式並凍結梯度
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # ========================================
        # 3. 初始化 Wavelet KD Loss
        # ========================================
        if self.config.get('kd_type', 'None') == 'Wavelet':
            from models.losses import WaveletKDLoss
            self.wavelet_criterion = WaveletKDLoss().cuda()
            print("✅ Wavelet KD Loss 引擎已啟動")

        # Optimizer & Scheduler
        self.optimizer_G = optim.Adam(self.netG.parameters(),
                                      lr=self.config['optimizer']['lr'],
                                      betas=(0.9, 0.999), eps=1e-8)
        self.scheduler_G = CosineAnnealingLR(self.optimizer_G,
                                             T_max=self.config['num_epochs'],
                                             eta_min=self.config['scheduler']['min_lr'])

    def _run_epoch(self, epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        epoch_size = self.config.get('train_batches_per_epoch', len(self.train_dataset))
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch: {}/{} | lr: {:.6f}'.format(
            epoch, self.config['num_epochs'], lr))

        self.netG.train()

        for i, data in enumerate(tq):
            inputs = data['a'].cuda()
            targets = data['b'].cuda()

            # --- 學生推論 ---
            outputs = self.netG(inputs)

            # --- 老師推論 (無梯度) ---
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)

            # --- Loss 計算開始 ---
            # 1. 原始 Stripformer 結構 Loss (含 Charbonnier + Edge 等)
            loss_original = self.criterionG(outputs, targets, inputs)

            # 2. 準備 KD Loss (預設為 0)
            loss_kd_output = torch.tensor(0.0).cuda()
            loss_kd_wavelet = torch.tensor(0.0).cuda()
            
            kd_type = self.config.get('kd_type', 'None')

            # 如果是 ExpB 或 ExpC，都會計算基礎的 L1 Output KD
            if kd_type in ['Output', 'Wavelet']:
                loss_kd_output = torch.nn.functional.l1_loss(outputs, teacher_outputs)

            # 只有 ExpC 才會計算 Wavelet KD
            if kd_type == 'Wavelet':
                loss_kd_wavelet = self.wavelet_criterion(outputs, teacher_outputs)

            # 3. 讀取權重並加總 Loss
            weight_kd_out = self.config.get('weight_kd_out', 0.5)
            weight_kd_wav = self.config.get('weight_kd_wav', 0.5)

            loss_total = loss_original + (weight_kd_out * loss_kd_output) + (weight_kd_wav * loss_kd_wavelet)

            # --- 反向傳播 ---
            self.optimizer_G.zero_grad()
            loss_total.backward()
            self.optimizer_G.step()

            # --- 指標記錄 ---
            self.metric_counter.add_losses(loss_total.item())
            curr_psnr, curr_ssim, _ = self.metric_counter.add_metrics(outputs, targets)
            tq.set_postfix(loss='{:.5f}'.format(self.metric_counter.loss_message()))

            wandb.log({
                "train/loss_original": loss_original.item(),
                "train/loss_kd_output": loss_kd_output.item(),
                "train/loss_kd_wavelet": loss_kd_wavelet.item(),
                "train/loss_total": loss_total.item(),
                "train/lr": lr
            })

            if i >= epoch_size:
                break
                
        tq.close()
        self.scheduler_G.step()

    def validate(self):
        self.netG.eval()
        self.metric_counter.clear()
        tq = tqdm.tqdm(self.val_dataset)
        tq.set_description('Validation')
        with torch.no_grad():
            for i, data in enumerate(tq):
                inputs = data['a'].cuda()
                targets = data['b'].cuda()
                outputs = self.netG(inputs)
                curr_psnr, curr_ssim, _ = self.metric_counter.add_metrics(outputs, targets)
                tq.set_postfix(psnr='{:.2f}'.format(curr_psnr), ssim='{:.4f}'.format(curr_ssim))
        tq.close()
        val_psnr, val_ssim = self.metric_counter.update_best_model()
        
        wandb.log({
            "val/psnr": val_psnr,
            "val/ssim": val_ssim
        })
        print(f"Validation PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")


if __name__ == '__main__':
    with open('config/config_Stripformer_gopro.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup WandB
    wandb.init(project='Uformer_Distill_Project', name=f"{config['experiment_desc']}")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Set random seed
    seed = 666
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    batch_size = config.pop('batch_size')
    get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=cpu_count(),
                             drop_last=True, pin_memory=True)

    train_dataset = PairedDataset(config['train'])
    val_dataset = PairedDataset(config['val'])

    train_loader = get_dataloader(train_dataset, shuffle=True)
    val_loader = get_dataloader(val_dataset, shuffle=False)

    trainer = Trainer(config, train_loader, val_loader)
    trainer.train()