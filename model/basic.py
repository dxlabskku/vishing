import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import argparse
import glob
import librosa


parser = argparse.ArgumentParser(description="PyTorch")
parser.add_argument('--wav_path', type=str, default='data/split_wav', help='wav_path')
parser.add_argument('--checkpoints_path', type=str,  help='checkpoints_path')
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

parser.add_argument('--model_name', type=str, default='SVM', help='model_name')

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

model_name = args.model_name


normal_path = glob.glob(f'{wav_path}/{feature_time}/normal/*.wav')
spam_path = glob.glob(f'{wav_path}/{feature_time}/spam/*.wav')
print(f'normal: {len(normal_path)} \t spam: {len(spam_path)}')

paths = np.concatenate([normal_path,spam_path])
normal_label = np.zeros(shape=(len(normal_path,)))
spam_label = np.ones(shape=(len(spam_path,)))
label = np.concatenate([normal_label,spam_label])


X_train, X_test, y_train, y_test = train_test_split(paths,label,stratify=label,test_size=0.2,random_state=42)
print(f'Train: {len(X_train)} \t Test: {len(X_test)}')

def SetData(paths, labels, n_fft, win_length, hop_length, n_mels, n_mfcc, feature_type):
    for idx,path in enumerate(paths):
        y, sr = librosa.load(path)
        D = np.abs(librosa.stft(y, n_fft=n_fft, win_length = win_length, hop_length=hop_length))

        if feature_type == 'stft':
            tmp = librosa.power_to_db(D, ref=np.max)
        elif feature_type == 'mel':
            mel_spec = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
            tmp = librosa.amplitude_to_db(mel_spec, ref=0.00002)
        elif feature_type == 'mfcc':
            mfcc = librosa.feature.mfcc(S = librosa.power_to_db(D), sr = sr, n_mfcc = n_mfcc)
            tmp = librosa.amplitude_to_db(mfcc, ref=0.00002)
        if idx == 0:
            shape = tmp.shape
            dataset = np.zeros((len(paths), shape[0]*shape[1]))
            labels_data = np.zeros((len(paths), ))
        tmp = tmp.reshape(1,-1)
        dataset[idx] = tmp
        labels_data[idx] = int(labels[idx])
    return dataset, labels

trainset, train_label = SetData(X_train, y_train, n_fft, win_length, hop_length, n_mels, n_mfcc, feature_type)
testset, test_label = SetData(X_test, y_test, n_fft, win_length, hop_length, n_mels, n_mfcc, feature_type)

if model_name == 'SVM':
    model = SVC(C = 10, kernel = 'linear')
    start = time.time()
    model.fit(trainset, train_label)
    train_time = time.time() - start
    print(f"training time : {train_time}")
    start_test = time.time()
    pred = model.predict(testset)
    test_time = time.time() - start_test
    print(f"test time : {test_time}")

    report = classification_report(pred, test_label, digits=6, output_dict = True)
    result = pd.DataFrame(report).transpose()
    result['train_time'] = train_time
    result['test_time'] = test_time
    result.to_csv(f'{result_path}/{feature_time}_{num_test}.csv')

elif model_name == 'Logistic':
    model = LogisticRegression(max_iter = 100, C = 100, penalty = 'l2', solver = 'lbfgs')
    start = time.time()
    model.fit(trainset, train_label)
    train_time = time.time() - start
    print(f"training time : {train_time}")
    start_test = time.time()
    pred = model.predict(testset)
    test_time = time.time() - start_test
    print(f"test time : {test_time}")

    report = classification_report(pred, test_label, digits=6, output_dict = True)
    result = pd.DataFrame(report).transpose()
    result['train_time'] = train_time
    result['test_time'] = test_time
    result.to_csv(f'{result_path}/{feature_time}_{num_test}.csv')

elif model_name == 'DT':
    model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 10)
    start = time.time()
    model.fit(trainset, train_label)
    train_time = time.time() - start
    print(f"training time : {train_time}")
    start_test = time.time()
    pred = model.predict(testset)
    test_time = time.time() - start_test
    print(f"test time : {test_time}")

    report = classification_report(pred, test_label, digits=6, output_dict = True)
    result = pd.DataFrame(report).transpose()
    result['train_time'] = train_time
    result['test_time'] = test_time
    result.to_csv(f'{result_path}/{feature_time}_{num_test}.csv')


elif model_name == 'RF':
    model = RandomForestClassifier(criterion = 'gini',max_depth = 20, n_estimators = 100)
    start = time.time()
    model.fit(trainset, train_label)
    train_time = time.time() - start
    print(f"training time : {train_time}")
    start_test = time.time()
    pred = model.predict(testset)
    test_time = time.time() - start_test
    print(f"test time : {test_time}")

    report = classification_report(pred, test_label, digits=6, output_dict = True)
    result = pd.DataFrame(report).transpose()
    result['train_time'] = train_time
    result['test_time'] = test_time
    result.to_csv(f'{result_path}/{feature_time}_{num_test}.csv')
