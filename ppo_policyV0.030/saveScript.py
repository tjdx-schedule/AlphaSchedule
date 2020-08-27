#import copy
#import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.game.envs import make_vec_envs
from a2c_ppo_acktr.game.ExcelLog import ExcelLog
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize
from evaluation import evaluate,save_model


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda:0" if args.cuda else "cpu")

    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                      args.gamma, args.log_dir, device_cpu, 'train')

    agent, ret_rms, log = \
        torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))#map_location=lambda storage, loc: storage
    actor_critic = agent.actor_critic
    log.setLog(args,args.excel_save)
    print('load model:', args.env_name, 'success!')
    actor_critic.to(device_gpu)


    save_model(args.save_dir,args.env_name,args.algo,
       actor_critic,ret_rms,
       None,log)


if __name__ == "__main__":
    main()