#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
#net_arg.add_argument('--input_scale_size', type=int, default=64, help='input image will be resized with the given value as width and height')
#net_arg.add_argument('--conv_hidden_num', type=int, default=128, choices=[64, 128,16,32],help='n in the paper')
net_arg.add_argument('--hidden',  type=str, default='5,5', help='comma separated list of hidden layer units')
#net_arg.add_argument('--varname', type=str, default='SPDT', help='names of trained variable')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='SPDT', help='names of trained variable')
data_arg.add_argument('--batch_size', type=int, default=2048)
data_arg.add_argument('--frac_train', type=float, default=0.8)
data_arg.add_argument('--local', type=str2bool, default=False)
data_arg.add_argument('--epoch', type=int, default=8)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--max_step', type=int, default=1000000)
train_arg.add_argument('--lr_update_step', type=int, default=100000, choices=[100000, 75000, 10000, 20000, 1000])
train_arg.add_argument('--lr', type=float, default=0.001)
train_arg.add_argument('--lr_lower_boundary', type=float, default=2e-6)
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--gamma', type=float, default=0.5)
train_arg.add_argument('--lambda_k', type=float, default=0.001)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)
train_arg.add_argument('--run_validation', type=str2bool, default=True)

# Misc
#parser.add('-c', '--config', default='', is_config_file=True, help='config file path')
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--save_step', type=int, default=5000)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--random_seed', type=int, default=123)

def get_config():
    config, unparsed = parser.parse_known_args()
    if config.use_gpu:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    setattr(config, 'data_format', data_format)
    return config, unparsed