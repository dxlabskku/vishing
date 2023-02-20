# feature extraction 학습시 함께 적용
import argparse
import numpy as np
import glob
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time
from sklearn.metrics import classification_report
import pandas as pd
import librosa


parser = argparse.ArgumentParser(description="PyTorch")
parser.add_argument('--wav_path', type=str, default='data/split_wav', help='wav_path')
parser.add_argument('--checkpoints_path', type=str, help='checkpoints_path')
parser.add_argument('--result_path', type=str,  help='result_path')

parser.add_argument('--learning_rate', default=1e-4, help='learning_rate')
parser.add_argument('--weight_decay', default=1e-4, help='weight_decay')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
parser.add_argument('--num_epoch', type=int, default=50, help='num_epoch')
parser.add_argument('--gpu_id', type=int, default=3, help='gpu_id')
parser.add_argument('--num_test', type=int, default=0, help='num_test')

parser.add_argument('--feature_type', type=str, default='stft', help='feature_type')
parser.add_argument('--feature_time', type=str, default='0.5', help='feature_time size')
parser.add_argument("--n_fft",default=4096,type=int)
parser.add_argument("--win_length",default=4096,type=int)
parser.add_argument("--hop_length",default=512,type=int)
parser.add_argument("--n_mels",default=256,type=int)
parser.add_argument("--n_mfcc",default=30,type=int)

args = parser.parse_args()

message = ''
message += '----------------- Options ---------------\n'
for k, v in sorted(vars(args).items()):
    comment = ''
    default = parser.get_default(k)
    if v != default:
        comment = '\t[default: %s]' % str(default)
    message += '{:>20}: {:<15}{}\n'.format(str(k), str(v), comment)
message += '----------------- End -------------------'
# print(message)



wav_path = args.wav_path
feature_type = args.feature_type
feature_time= args.feature_time
learning_rate = args.learning_rate
weight_decay = args.weight_decay
batch_size = args.batch_size
num_epoch = args.num_epoch
checkpoints_path = args.checkpoints_path
result_path = args.result_path
num_test = args.num_test

n_fft = args.n_fft
win_length = args.win_length
hop_length = args.hop_length
n_mels = args.n_mels
n_mfcc = args.n_mfcc

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")


normal_path = glob.glob(f'{wav_path}/{feature_time}/normal/*.wav')
spam_path = glob.glob(f'{wav_path}/{feature_time}/spam/*.wav')
print(f'normal: {len(normal_path)} \t spam: {len(spam_path)}')

paths = np.concatenate([normal_path,spam_path])
normal_label = np.zeros(shape=(len(normal_path,)))
spam_label = np.ones(shape=(len(spam_path,)))
label = np.concatenate([normal_label,spam_label])


X_train, X_test, y_train, y_test = train_test_split(paths,label,stratify=label,test_size=0.2,random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,stratify=y_train,test_size=0.16,random_state=42)
print(f'Train: {len(X_train)} \t Val: {len(X_valid)} \t Test: {len(X_test)}')

def SetData(paths, labels, n_fft, win_length, hop_length, n_mels, n_mfcc, feature_type):
    dataset = []
    for idx,path in enumerate(paths):
        y, sr = librosa.load(path)

        D = np.abs(librosa.stft(y, n_fft=n_fft, win_length = win_length, hop_length=hop_length))
        # print(f'        Wav -> features: {n+1} / {len(wav_list)}')

        if feature_type == 'stft':
            tmp = librosa.power_to_db(D, ref=np.max)
        elif feature_type == 'mel':
            mel_spec = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
            tmp = librosa.amplitude_to_db(mel_spec, ref=0.00002)
        elif feature_type == 'mfcc':
            mfcc = librosa.feature.mfcc(S = librosa.power_to_db(D), sr = sr, n_mfcc = n_mfcc)
            tmp = librosa.amplitude_to_db(mfcc, ref=0.00002)

        tmp = np.expand_dims(tmp, axis = 0)
        tmp = torch.tensor(tmp)
        tmp_y = int(labels[idx])
        dataset.append((tmp, tmp_y))
    return dataset

trainset = SetData(X_train, y_train, n_fft, win_length, hop_length, n_mels, n_mfcc, feature_type)
validset = SetData(X_valid, y_valid, n_fft, win_length, hop_length, n_mels, n_mfcc, feature_type)
testset = SetData(X_test, y_test, n_fft, win_length, hop_length, n_mels, n_mfcc, feature_type)
batch_size = 32
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), num_workers=0, shuffle = False)

class LSTM(nn.Module):
    def __init__(self, device, input_dim, hidden_dim):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.build_model()
        self.to(device)
        
    
    def build_model(self):
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers = 3, 
                            dropout = 0.2, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(2*self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.BCE_loss = nn.BCELoss()
    
    def forward(self, x):
        output, _ = self.lstm(x)
        output = torch.mean(output, dim = 1)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output.squeeze()
    
    def train(self, train_loader, valid_loader, epochs, learning_rate):
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        
        loss_log = []
        for e in range(epochs):
            epoch_loss = 0
            for _, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                data = data.squeeze()
                data, target = data.to(self.device), target.to(device=self.device, dtype=torch.float32)
                out = self.forward(data)
                loss = self.BCE_loss(out, target)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                
            loss_log.append(epoch_loss)
            
            valid_acc, valid_loss = self.predict(valid_loader)
            print(f'>> [Epoch {e+1}/{epochs}] Total epoch loss: {epoch_loss:.2f} / Valid accuracy: {100*valid_acc:.2f}% / Valid loss: {valid_loss:.4f}')
        return loss_log
    
    def predict(self,valid_loader, return_preds = False):
        BCE_loss = nn.BCELoss(reduction = 'sum')
        preds = []
        total_loss = 0
        correct = 0
        len_data = 0
        with torch.no_grad():
            for _, (data, target) in  enumerate(valid_loader):
                data = data.squeeze()
                data, target = data.to(self.device), target.to(device=self.device, dtype=torch.float32)
                out = self.forward(data)
                len_data += len(target)
                loss = BCE_loss(out, target)
                total_loss += loss
                
                pred = (out>0.5).detach().cpu().numpy().astype(np.float32)
                preds += list(pred)
                correct += sum(pred == target.detach().cpu().numpy())
                
            acc = correct / len_data
            loss = total_loss/len_data
            
        if return_preds:
            return acc, loss, preds
        else:
            return acc, loss
          

print(device)
learning_rate = 1e-4

input_dim = trainset[0][0].shape[-1]
hidden_dim = 16
model = LSTM(device = device, input_dim=input_dim, hidden_dim=hidden_dim)
model.to(device)

start = time.time()
model.train(train_loader = train_loader, valid_loader = valid_loader, epochs = num_epoch, learning_rate = learning_rate)
train_time = time.time() - start
print(f'Train time: {train_time}')
torch.save(model.state_dict(), f'{checkpoints_path}/{feature_time}_{num_test}.pt')

start_t = time.time()
acc, loss, p = model.predict(test_loader, return_preds = True)
test_time = time.time() - start_t
print(f'Test time: {test_time}')
report = classification_report(y_test,p,digits=6, output_dict = True)

result = pd.DataFrame(report).transpose()
result['train_time'] = train_time
result['test_time'] = test_time
result.to_csv(f'{result_path}/{feature_time}_{num_test}.csv')
