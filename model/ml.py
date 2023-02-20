import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--feature_type', type=str, default='mfcc', help='feature_type')
parser.add_argument('--feature_time', type=float, default=0.5, help='feature_time')
parser.add_argument('--model_name', type=str, default='DenseNet', help='model_name')
parser.add_argument('--wav_path', type=str, help='wav_path')
parser.add_argument('--checkpoints_path', type=str, help='checkpoints_path')
parser.add_argument('--result_path', type=str, help='result_path')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
args = parser.parse_args()


result_model_path = f"{result_root}/{model}"
checkpoints_model_path = f"{checkpoints_root}/{model}"
if not os.path.exists(result_model_path):
    os.mkdir(result_model_path)
if not os.path.exists(checkpoints_model_path):
    os.mkdir(checkpoints_model_path)
  
result_path = f"{result_model_path}/{feature}"
checkpoints_path = f"{checkpoints_model_path}/{feature}"
if not os.path.exists(result_path):
    os.mkdir(result_path)
if not os.path.exists(checkpoints_path):
    os.mkdir(checkpoints_path)


os.system(f'python basic.py --model_name {args.model_name} --feature_time {args.feature_time} --feature_type {args.feature_type} --checkpoints_path {args.checkpoints_path} --result_path {args.result_path} --wav_path {args.num_test}')


print("FINISH")
