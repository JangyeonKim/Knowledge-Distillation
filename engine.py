import torch
import torch.nn as nn
import pytorch_lightning as pl

from sklearn.metrics import accuracy_score, confusion_matrix
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import numpy as np 

import config_s # student config
import config # teacher config

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import pdb

# lam and tem for distillation
lam = config_s.lam
tem = config_s.tem

device = config_s.device

spectrogram_extractor = Spectrogram(n_fft=config_s.window_size, hop_length=config_s.hop_size, win_length=config_s.window_size, window=config_s.window, center=config_s.center, pad_mode=config_s.pad_mode, freeze_parameters=True).to(device)
logmel_extractor = LogmelFilterBank(sr=config_s.sample_rate, n_fft=config_s.window_size, n_mels=config_s.mel_bins, fmin=config_s.fmin, fmax=config_s.fmax, ref=config_s.ref, amin=config_s.amin, top_db=config_s.top_db, freeze_parameters=True).to(device)

class Engine(pl.LightningModule) :
    def __init__(self, model) :
        super().__init__()
        self.model = model
    
    def val_evaluate_metric(self, pred, ans):
        acc = accuracy_score(ans, np.argmax(pred, 1))
        return {"val_acc": acc} 
    
    def test_evaluate_metric(self, pred, ans):
        acc = accuracy_score(ans, np.argmax(pred, 1))
        return {"test_acc": acc} 
    
    def forward(self, x) :
        return self.model(x)
    
    def training_step(self, batch, batch_idx) :
        x, y = batch["waveform"].to(device), batch["target"].to(device)
        spec_x = spectrogram_extractor(x)
        mel_spec_x = logmel_extractor(spec_x)
        mel_spec_x = mel_spec_x.repeat(1, 3, 1, 1)
        y_hat = self.model(mel_spec_x)
        # pdb.set_trace()
        
        # loss = nn.CrossEntropyLoss()(y_hat, y)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=config_s.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx) :
        x, y = batch["waveform"].to(device), batch["target"].to(device)
        spec_x = spectrogram_extractor(x)
        mel_spec_x = logmel_extractor(spec_x)
        mel_spec_x = mel_spec_x.repeat(1, 3, 1, 1)
        y_hat = self.model(mel_spec_x)
        
        # loss = nn.CrossEntropyLoss()(y_hat, y)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=config_s.batch_size)
        
        return [y_hat.detach(), y.detach()]
    
    def validation_epoch_end(self, validation_step_outputs) :
        y_hat = torch.cat([x[0] for x in validation_step_outputs], dim=0)
        y = torch.cat([x[1] for x in validation_step_outputs], dim=0)
        
        gather_pred = y_hat.cpu().numpy()
        gather_target = y.cpu().numpy()
        gather_target = np.argmax(gather_target, 1)
        
        metric_dict = self.val_evaluate_metric(gather_pred, gather_target)
        self.log("val_acc", metric_dict["val_acc"], on_epoch=True, prog_bar=True, logger=True, batch_size=config_s.batch_size)
        
    def test_step(self, batch, batch_idx) :
        x, y = batch["waveform"].to(device), batch["target"].to(device)
        spec_x = spectrogram_extractor(x)
        mel_spec_x = logmel_extractor(spec_x)
        mel_spec_x = mel_spec_x.repeat(1, 3, 1, 1)
        y_hat = self.model(mel_spec_x)
        
        return [y_hat.detach(), y, batch['audio_name']]
    
    def test_epoch_end(self, test_step_outputs):
        pred = torch.cat([d[0] for d in test_step_outputs], dim = 0)
        target = torch.cat([d[1] for d in test_step_outputs], dim = 0)
        audio_name = [name for sublist in [d[2] for d in test_step_outputs] for name in sublist]
        
        gather_pred = pred.cpu().numpy()
        gather_target = target.cpu().numpy()
        gather_target = np.argmax(gather_target, 1)
        
        # pdb.set_trace()
        
        metric_dict = self.test_evaluate_metric(gather_pred, gather_target)
        
        # ================== save csv ====================
        
        df = pd.DataFrame()
        df['file'] = audio_name
        df['model_output'] = gather_pred.tolist()
        df['model_prediction'] = np.argmax(gather_pred, 1).tolist()
        df['label'] = gather_target.tolist()
        
        save_result_dir = os.path.join(config.test_result_dir, config.exp_name)
        if not os.path.exists(save_result_dir):
            os.makedirs(save_result_dir)
        
        save_csv = os.path.join(save_result_dir , f'acc_[{metric_dict["test_acc"]*100:.2f}].csv')
        df.to_csv(save_csv , index = True)

        # ================== save confusion matrix ====================

        cm = confusion_matrix(gather_target, np.argmax(gather_pred, 1))
        labels = config.confusion_labels
        plt.figure(figsize=(7, 7))
        sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap = 'YlGnBu', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Label')
        
        save_confusion = os.path.join(save_result_dir , f'confusion_matrix_[{metric_dict["test_acc"]*100:.2f}].png')
        plt.savefig(save_confusion)
        
    def configure_optimizers(self) :
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
class DistillEngine(Engine) :
    def __init__(self, student_model, teacher_model) :
        super(DistillEngine, self).__init__(model = student_model)
        self.teacher_model = teacher_model
        # ********* teacher model load *********
        ckpt = torch.load(config.test_checkpoint, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        self.teacher_model.load_state_dict(ckpt["state_dict"], strict=False)
        print("teacher model loaded!")
        # # ********* teacher model eval *********
        # self.teacher_model.eval()
        # for param in self.teacher_model.parameters() :
        #     param.requires_grad = False
            
    def training_step(self, batch, batch_idx) :
        x, y = batch["waveform"].to(device), batch["target"].to(device)
        spec_x = spectrogram_extractor(x)
        mel_spec_x = logmel_extractor(spec_x)
        mel_spec_x = mel_spec_x.repeat(1, 3, 1, 1)
        y_hat_student = self.model(mel_spec_x)
        
        # ********* teacher model eval *********
        self.teacher_model.eval()
        for param in self.teacher_model.parameters() :
            param.requires_grad = False

        y_hat_teacher = self.teacher_model(x)
        y_hat_teacher = torch.sigmoid(y_hat_teacher['clipwise_output'] / tem) # already sigmoided in original HTS-AT code, but not in this code because of temperature for distillation
                                                                              # the more temperature, the more soft the output of teacher model
        soft_target_loss = nn.BCEWithLogitsLoss()(y_hat_student, y_hat_teacher)
        label_loss = nn.BCEWithLogitsLoss()(y_hat_student, y)
        
        loss = lam * label_loss + (1 - lam) * soft_target_loss
        
        
        self.log("distillation_loss", soft_target_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=config_s.batch_size)
        self.log("label_loss", label_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=config_s.batch_size)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=config_s.batch_size)
        
        return loss
    