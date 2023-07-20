import argparse
import config
import torch
import sys
import importlib
import numpy as np
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import lightning_fabric as lf

from engine import DistillEngine, Engine
from dataset import JY_Dataset
from models.mobileNet import mobilenet_v3_large, mobilenet_v3_small
from models.htsat import HTSAT_Swin_Transformer

import config
import config_s

parser = argparse.ArgumentParser() 
parser.add_argument('--mode', type=str, default='train')
args = parser.parse_args()

model_dict = {
        "mobilenet_v3_large" : mobilenet_v3_large(config_s),
        "mobilenet_v3_small" : mobilenet_v3_small(config_s),
}

def train() :
    lf.utilities.seed.seed_everything(config_s.random_seed)
    
    train_dataset = JY_Dataset(dataset = np.load(os.path.join(config_s.dataset_path, "train.npy"), allow_pickle = True), config = config_s)
    val_dataset = JY_Dataset(dataset = np.load(os.path.join(config_s.dataset_path, "val.npy"), allow_pickle = True), config = config_s)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config_s.batch_size, shuffle=True, num_workers=config_s.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config_s.batch_size, shuffle=False, num_workers=config_s.num_workers)

    model = model_dict[config_s.model_name]
    sed_model =  HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config = config,
        depths = config.htsat_depth,
        embed_dim = config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )
        
    engine = DistillEngine(student_model=model, teacher_model=sed_model)
    
    checkpoint_callback = ModelCheckpoint(
    save_top_k = 10,
    monitor = "val_loss",
    mode = "min",
    filename = "{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
    )
    
    logger = TensorBoardLogger(config_s.save_path, name = config_s.exp_name)
    
    trainer = pl.Trainer(
        deterministic=False,
        default_root_dir = config_s.save_path,
        devices = config_s.num_gpu, 
        val_check_interval = 0.1,
        max_epochs = config_s.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        callbacks = [checkpoint_callback],
        accelerator = "gpu",
        num_sanity_val_steps = 0,
        resume_from_checkpoint = None, 
        replace_sampler_ddp = False,
        gradient_clip_val=1.0,
        logger = logger
    )

    
    # ********* teacher model load *********
    ckpt = torch.load(config.test_checkpoint, map_location="cpu")
    ckpt["state_dict"].pop("sed_model.head.weight")
    ckpt["state_dict"].pop("sed_model.head.bias")
    
    new_ckpt_state_dict = {}
    for key,value in ckpt["state_dict"].items() :
        new_key = key.replace("sed_model.", "")
        new_ckpt_state_dict[new_key] = value
    
    sed_model.load_state_dict(new_ckpt_state_dict, strict=False)
    print("teacher model loaded!")
    
    trainer.fit(engine, train_dataloader, val_dataloader)
    
def test() : 
    lf.utilities.seed.seed_everything(config_s.random_seed)
    
    test_dataset = JY_Dataset(dataset = np.load(os.path.join(config_s.dataset_path, "test.npy"), allow_pickle = True), config = config_s)
    test_dataloader = DataLoader(test_dataset, batch_size=config_s.batch_size, shuffle=False, num_workers=config_s.num_workers)
    
    model = model_dict[config_s.model_name]
    
    engine = Engine(model)

    trainer = pl.Trainer(
    deterministic=False,
    devices = config_s.num_gpu, 
    auto_lr_find = True,    
    sync_batchnorm = True,
    checkpoint_callback = False,
    accelerator = "gpu" ,
    num_sanity_val_steps = 0,
    replace_sampler_ddp = False,
    gradient_clip_val=1.0
    )
    

    
    if config_s.checkpoint_path is not None :
        ckpt = torch.load(config_s.checkpoint_path, map_location="cpu")
        new_ckpt_state_dict = {}
        for key,value in ckpt["state_dict"].items() :
            if "teacher_model" not in key :
                new_key = key.replace("model.", "")
                new_ckpt_state_dict[new_key] = value
        
        # import pdb; pdb.set_trace()
        model.load_state_dict(new_ckpt_state_dict)
        # import pdb; pdb.set_trace()
        # trainer.test(model = engine, test_dataloaders = test_dataloader, ckpt_path = config_s.checkpoint_path)
        trainer.test(model = engine, test_dataloaders = test_dataloader)
        print("checkpoint loaded")
    else :
        print("checkpoint not loaded")
        sys.exit(1)
    
if __name__ == "__main__" :
    if args.mode == "train" :
        train()
    elif args.mode == "test" :
        test()
    else :
        print("Wrong mode")
        sys.exit(1)