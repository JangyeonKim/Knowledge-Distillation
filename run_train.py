import subprocess

lamd = [0.5, 0.3, 0.1, 0.0] 
temp = [1.0, 3.0]

for t in temp :
    for l in lamd :
        with open("config_s_tmp.py", "w") as file :
            file.write(f"""
import torch
import os

model_name = "mobilenet_v3_small"

test_result_dir = "/home/jykim/work/MobileNetV3-Large/csvs & confusion matrix"
checkpoint_path = "/home/jykim/work/MobileNetV3-Large/result/lightning_logs/efficientnet_v2_s/checkpoints/epoch=18-val_loss=0.0028-val_acc=0.9982.ckpt"

confusion_labels = ['Vehicle', 'Footsteps', 'Other']

lam = {l} # 0.5, 0.3, 0.1, 0.0 
tem = {t} # 1.0 3.0

exp_name = f"V3s_lam_{l}_tem_{t}" # the name of your experiment
                                      # train < distillation hyperparameter
                                      # test < checkpoint_path.split('/')[-1][:-5]

num_gpu = [0]
os.environ["CUDA_VISIBLE_DEVICES"] =f"0"
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
max_epoch = 30

random_seed = 42

classes_num = 3
batch_size = 128 # MN : 256 # e-b0,b1,b2: 128 # e-v2-s, b3 : 64
num_workers = 8
####################
dataset_path = "/home/jykim/work/HTS-AT/aihub_dataset/v2_MotorAndCar"
save_path = "/home/jykim/work/Knowledge-Distillation/result"
####################

# for signal processing
sample_rate = 16000 # 16000 for scv2, 32000 for audioset and esc-50
clip_samples = sample_rate * 10 # audio_set 10-sec clip
window_size = 1024
hop_size = 160 # 160 for scv2, 320 for audioset and esc-50
mel_bins = 64
fmin = 50
fmax = 14000
####################
window = 'hann'
center = True
pad_mode = 'reflect'
ref = 1.0
amin = 1e-10
top_db = None
"""
)
        subprocess.call(["mv", "config_s_tmp.py", "config_s.py"])
        subprocess.call(["python", "main.py", "--mode", "train"])