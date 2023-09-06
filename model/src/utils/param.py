import argparse, random, torch
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--splitRate", dest='split_rate', type=float, default=None)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--aLearning", action='store_true')
    parser.add_argument("--pred", action='store_true')

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=200)
    parser.add_argument('--predNum', dest='pred_num', type=int, default=100)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--samples_num', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # IO
    parser.add_argument('--output', type=str, default = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'outputs')) # ! Default for debug
    parser.add_argument('--input', type=str, default= os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'inputs')) # ! Default for debug

    # Model Loading
    parser.add_argument('--load', type=str, default=None, help='Load specified weights, usually the fine-tuned weights for our task for testing.')

    # Training configuration
    parser.add_argument("--multiGPU", dest='multi_GPU', action='store_true')
    parser.add_argument("--numWorkers", dest='num_workers', default=0)

    # Parse the arguments.
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()