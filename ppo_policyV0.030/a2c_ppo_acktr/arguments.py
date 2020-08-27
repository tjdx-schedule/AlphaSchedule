import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: ppo')
    parser.add_argument(
        '--lr', type=float, default=2.5e-4, help='learning rate (default: 2.5e-4)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=1,#0.99,
        help='discount factor for rewards (default: 1)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--use-gae',
        type=bool,
        # action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=8,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=1024,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=16,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.1,
        help='ppo clip parameter (default: 0.1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='log interval, one log per n updates (default: 1)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--load',
        action='store_true',
        default=False,
        help='load trained some model to train more')
    parser.add_argument(
        '--excel-save',
        action='store_true',
        default=False,
        help='is save statistic to excel')
    parser.add_argument(
        '--load-dir',
        default='./trained_models/ppo',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--env-name',
        default='ml-agent',
        help='environment to train on (default: ml-agent)')
    parser.add_argument(
        '--env-mode',
        default= None,
        help='choose ZhangeEnv state is value or canvas')
    parser.add_argument(
        '--eval-num',
        type=int,
        default=96,
        help='test number')
    parser.add_argument(
        '--log-dir',
        default='./logs/gym/',
        help='directory to save agent logs (default: ./logs/gym/)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--recurrent-accelerate',
        action='store_true',
        default=False,
        help='train a recurrent policy model use accelerate ways,when recurrent is true')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
