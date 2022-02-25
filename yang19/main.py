import os
from models import RNNNet

import argparse
import numpy as np
import random
from pathlib import Path

from make_environments import *
from train import main as train
from analyze import main as analyze
from analyze import print_performance

import logging
_logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='nr. of hidden units')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='sequence length')
    parser.add_argument('--dt', type=int, default=100,
                        help='neuronal time constant')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--sigma_rec', type=int, default=0.05,
                        help='recurrent unit noise')

    parser.add_argument('--cuda', action='store_true',
                        help='if set, will run with cuda')

    args = parser.parse_args()

    _logger.info("Running with args %s", vars(args))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _logger.info(f'Device: {device}')

    # create save directory
    path = Path('.') / 'figures'
    os.makedirs(path, exist_ok=True)
    path = Path('.') / 'files'
    os.makedirs(path, exist_ok=True)

    _logger.info('********* TRAINING **********')
    env, dataset, ob_size, act_size = make_train_environment(args)
    model = RNNNet(input_size=ob_size, hidden_size=args.hidden_size, output_size=act_size,
                   dt=env.dt).to(device)
    print(model)
    train(args, model, device, env, dataset, act_size)

    _logger.info('********* ANALYZING **********')
    env = make_analyze_environment(args)
    tasks = ngym.get_collection('yang19')
    #task variance
    analyze(args, model, env, tasks, device)

if __name__ == '__main__':
    main()



