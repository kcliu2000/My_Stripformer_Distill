import os
files_to_patch = ['train_Stripformer_pretrained.py', 'train_Stripformer_gopro.py']

for file_name in files_to_patch:
    if not os.path.exists(file_name): continue
    with open(file_name, 'r') as f:
        code = f.read()

    if 'import wandb' in code:
        continue # 代表已經修改過了

    # 1. 加入套件
    code = 'import wandb\nimport numpy as np\n' + code
    
    # 2. 初始化 WandB (綁定到你的專案，並加上 Stripformer 前綴)
    code = code.replace("torch.backends.cudnn.enabled = True", 
                        "wandb.init(project='Uformer_Distill_Project', name=f\"Stripformer_{config['experiment_desc']}\")\n    torch.backends.cudnn.enabled = True")
    
    # 3. 記錄訓練 Loss 到 WandB
    code = code.replace("self.metric_counter.write_to_tensorboard(epoch)", 
                        "self.metric_counter.write_to_tensorboard(epoch)\n        train_loss = np.mean(self.metric_counter.metrics['G_loss'])\n        wandb.log({'Train/G_loss': train_loss, 'epoch': epoch})")
    
    # 4. 印出驗證分數並上傳 WandB
    code = code.replace("self.metric_counter.write_to_tensorboard(epoch, validation=True)", 
                        "self.metric_counter.write_to_tensorboard(epoch, validation=True)\n        val_psnr = np.mean(self.metric_counter.metrics['PSNR'])\n        val_ssim = np.mean(self.metric_counter.metrics['SSIM'])\n        print(f'\\n🌊 [Epoch {epoch}] Validation PSNR: {val_psnr:.4f} | SSIM: {val_ssim:.4f}\\n')\n        wandb.log({'Val/PSNR': val_psnr, 'Val/SSIM': val_ssim, 'epoch': epoch})")

    with open(file_name, 'w') as f:
        f.write(code)
print("✨ Stripformer 的兩個訓練檔都已成功注入 WandB 與終端機顯示功能！")