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

    parser.add_argument('--training', type=int, default=40000,
                        help='number of training iterations')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='training batch size')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='sequence length')
    parser.add_argument('--dt', type=int, default=100,
                        help='neuronal time constant')

    #parser.add_argument('--seed', type=int, default=42,
    #                    help='random seed')
    parser.add_argument('--seeds', nargs='+', default=['42'], help='List of seeds')
    parser.add_argument('--sigma_rec', type=int, default=0.05,
                        help='recurrent unit noise')

    parser.add_argument('--cuda', action='store_true',
                        help='if set, will run with cuda')

    parser.add_argument('--dryrun', action='store_true',
                        help='if set, will override number of training')

    args = parser.parse_args()
    _logger.info("Running with args %s", vars(args))
    seeds = [int(x) for x in args.seeds]
    print(seeds)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _logger.info(f'Device: {device}')

    if args.dryrun:
        args.training = 100

    # create save directory
    path = Path('.') / 'figures'
    os.makedirs(path, exist_ok=True)
    path = Path('.') / 'files'
    os.makedirs(path, exist_ok=True)

    for seed in seeds:
        _logger.info('********* TRAINING **********')
        env, dataset, ob_size, act_size = make_train_environment(args, seed)
        model = RNNNet(input_size=ob_size, hidden_size=args.hidden_size, output_size=act_size,
                       sigma_rec=args.sigma_rec, dt=env.dt).to(device)
        print(model)
        train(args, seed, model, device, env, dataset, act_size)

        _logger.info('********* ANALYZING **********')
        env = make_analyze_environment(args, seed)
        tasks = ngym.get_collection('yang19')
        #task variance
        analyze(args, seed, model, env, tasks, device)

if __name__ == '__main__':
    main()



