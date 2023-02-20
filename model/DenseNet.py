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
parser.add_argument('--result_path', type=str, help='result_path')

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
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=0, shuffle = False)

class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channels = 4*growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn2 = nn.BatchNorm2d(inner_channels)
        self.conv2 = nn.Conv2d(inner_channels, growth_rate, kernel_size = 3, stride = 1, padding = 1, bias = False)
        
    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out =self.conv2(out)
        out = torch.cat((x, out), 1)
        return out    
      
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        
    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, kernel_size =2, stride = 2)
        return out
      
class DenseNet(nn.Module):
    def __init__(self, growth_rate, nblocks, reduction, num_classes, init_weights = True):
        super().__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, inner_channels, kernel_size = 7, stride = 2, padding =3),
            nn.MaxPool2d(3, stride = 2, padding = 1)
        )
        
        self.features = nn.Sequential()
        
        for i in range(len(nblocks)-1):
            self.features.add_module('dense_block_{}'.format(i), self._make_dense_block(nblocks[i], inner_channels))
            inner_channels += self.growth_rate * nblocks[i]
            out_channels = int(reduction*inner_channels)
            self.features.add_module('transition_{}'.format(i), Transition(inner_channels, out_channels))
            inner_channels = out_channels
        
        self.features.add_module('dense_block_{}'.format(len(nblocks)-1), self._make_dense_block(nblocks[len(nblocks)-1], inner_channels))
        inner_channels += self.growth_rate * nblocks[len(nblocks)-1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU())
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(inner_channels, num_classes)
        
        if init_weights:
            self._initialize_weights()
            
    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.sigmoid(out)
        return out
                
    
    def _make_dense_block(self, nblock, inner_channels):
        layers = []
        for _ in range(nblock):
            layers.append(BottleNeck(inner_channels, self.growth_rate))
            inner_channels += self.growth_rate
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)   

# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)
    

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()
    
    return loss_b.item(), metric_b                

# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = 0

    for xb, yb in dataset_dl:
        xb = torch.tensor(xb)
        xb = xb.to(device)
        yb = torch.tensor(yb)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b
        len_data += len(yb)
        
        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break
    
    loss = running_loss / len_data
    metric = running_metric / len_data
    return loss, metric


# function to start training
def train_val(model, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    opt=params['optimizer']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    sanity_check=params['sanity_check']
    lr_scheduler=params['lr_scheduler']
    path2weights=params['path2weights']

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr= {}'.format(epoch+1, num_epochs, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
#             print('Copied best model weights!')

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
#             print('Loading best model weights!')
            model.load_state_dict(best_model_wts)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, val_metric*100, (time.time()-start_time)/60))
        print('-'*10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


model = DenseNet(growth_rate=12, nblocks=[6,12,24,6], reduction = 0.5, num_classes=2)
model.to(device)

# define loss function, optimizer, lr_scheduler
loss_func = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=8)

params_train = {
    'num_epochs':num_epoch,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_loader,
    'val_dl':valid_loader,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt',
}


start = time.time()
model, loss_hist, metric_hist = train_val(model, params_train)
train_time = time.time() - start
print(f'Train time: {train_time}')

torch.save(model.state_dict(), f'{checkpoints_path}/{feature_time}_{num_test}.pt')


pred = []
start_t  = time.time()
for i, t in enumerate(test_loader):
    test_data = t
    data = test_data[0].to(device)
    output = model(data)
    output = output.detach().cpu().numpy()
    output = output.argmax(axis = 1)
    pred = pred +list(output)
test_time = time.time() - start_t
print(f'Test time: {test_time}')

report = classification_report(y_test,pred,digits=6, output_dict = True)

result = pd.DataFrame(report).transpose()
result['train_time'] = train_time
result['test_time'] = test_time
result.to_csv(f'{result_path}/{feature_time}_{num_test}.csv')
