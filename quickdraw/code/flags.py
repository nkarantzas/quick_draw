import torch
import argparse
from glob import glob
import os
from pathlib import Path

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Quick Draw")
#################################################################
# Data set options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_data', type=str,
                           default=str(Path(__file__).parents[2]) + '/train_simplified/')
group_dataset.add_argument('--test_data', type=str,
                           default=str(Path(__file__).parents[2]) + '/test_simplified/')
group_dataset.add_argument('--save_path', type=str, default=str(Path(__file__).parents[1]))
group_dataset.add_argument('--input_channels', type=int, default=1)
group_dataset.add_argument('--image_shape', type=int, default=(128, 128))
group_dataset.add_argument('--num_classes', type=int,
                           default=len(list(glob(parser.parse_known_args()[0].training_data + '*.csv'))))
#################################################################
# model training options
group_train = parser.add_argument_group('Training')
batch_size = 64 if torch.cuda.device_count() == 0 else torch.cuda.device_count() * 64
group_train.add_argument('--batch_size', type=int, default=batch_size)
group_train.add_argument('--val_samples_per_class', type=int, default=30)
group_train.add_argument('--num_epochs', type=int, default=1)
group_train.add_argument('--learning_rate', type=float, default=.001)
# how ofter do you want to print training and validation accuracies/losses
group_train.add_argument('--log_interval', type=int, default=100)
#################################################################
FLAGS, _ = parser.parse_known_args()

print(str(Path(__file__).parents[1]))

