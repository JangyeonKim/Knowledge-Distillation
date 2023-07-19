import subprocess
import os
import glob

path = "/home/jykim/work/Knowledge-Distillation/result/*/*/*/*.ckpt"

checkpoint_path_list = glob.glob(path)

for checkpoint in checkpoint_path_list :
    exp_name = checkpoint.split('/')[-4] + "-" + checkpoint.split('/')[-1][:-5]
    with open("config_s_tmp.py", "w") as file :
        file.write(f"""
import torch
import os

model_name = "mobilenet_v3_large"

test_result_dir = "/home/jykim/work/Knowledge-Distillation/csvs & confusion matrix"
checkpoint_path = "{checkpoint}"

confusion_labels = ['Vehicle', 'Footsteps', 'Other']

lam = None # 0.5, 0.3, 0.1, 0.0 
tem = None # 1.0 3.0

exp_name = f"{exp_name}" # the name of your experiment
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
        # subprocess.call(["mv", "config_s_tmp.py", "config_s.py"])
        # subprocess.call(["python", "main.py", "--mode", "test"])