import os 
import soundfile as sf
import librosa 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,help='data directory')
parser.add_argument('--target_root', type=str, help='target root directory')
opt = parser.parse_args()

data_path = opt.data_path
normal_path = f'{data_path}/normal'
normal_list = os.listdir(normal_path)

spam_path = f'{data_path}/spam'
spam_list = sorted(os.listdir(spam_path))

target_root = opt.target_root
if not os.path.exists(target_root):
    os.mkdir(target_root)
    
time_list = [0.5, 0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]

for split_time in time_list:
    print(f'\n================== {split_time}sec split ==================')
    target_root = f'{opt.target_root}/{split_time}'
    if not os.path.exists(target_root):
        os.mkdir(target_root)
    
    ## normal
    print('\nNormal data')
    targetdir = os.path.join(target_root, 'normal')
    if not os.path.exists(targetdir):
        os.mkdir(targetdir)        
    i = 0
    for num, wav in enumerate(normal_list):
        print(f'split: {i+1} / {len(normal_list)}')
        wav_file_path = f'{normal_path}/{wav}'
        # print(wav_file_path)
        y,sr = librosa.load(wav_file_path) 
        cur_sample = y[:int((split_time)*sr)]
        sf.write(f'{targetdir}/{i}_{wav[:-4]}.wav',cur_sample,sr,subtype='PCM_16') 
        i += 1
    

    ## spam
    print('\nspam data')
    targetdir = os.path.join(target_root, 'spam')
    if not os.path.exists(targetdir):
        os.mkdir(targetdir)    
    i = 0
    for num, wav in enumerate(spam_list):
        print(f'split: {i+1} / {len(spam_list)}')
        wav_file_path = f'{spam_path}/{wav}'
        # print(wav_file_path)
        y,sr = librosa.load(wav_file_path) 
        cur_sample = y[:int((split_time)*sr)]
        sf.write(f'{targetdir}/{i}_{wav[:-4]}.wav',cur_sample,sr,subtype='PCM_16') 
        i += 1
            
