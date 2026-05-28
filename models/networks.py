import torch.nn as nn
from models.Stripformer import Stripformer

def get_generator(model_config):
    generator_name = model_config.get('g_name', 'Stripformer')
    
    # 👇 從 yaml 讀取我們設定的層數，如果沒寫就預設為滿血版 12 層
    num_blocks = model_config.get('num_blocks', 12) 
    
    if generator_name == 'Stripformer':
        model_g = Stripformer(num_blocks=num_blocks) # 👈 將層數傳入
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    return nn.DataParallel(model_g)

def get_nets(model_config):
    return get_generator(model_config)