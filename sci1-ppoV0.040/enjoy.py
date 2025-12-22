import os
import sys
import time

import numpy as np
import torch

from a2c_ppo_acktr.game.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize
from a2c_ppo_acktr.algo import PPO
from a2c_ppo_acktr.game.ExcelLog import ExcelLog 
from a2c_ppo_acktr.game.EnvirConf import config
from a2c_ppo_acktr.arguments import argsECInit

sys.path.append('a2c_ppo_acktr')


args = argsECInit(isTrain = False)
args.det = not args.non_det
args.cuda = args.gpu > -1 and torch.cuda.is_available()
args.eval_load = not args.not_eval_load
#args.env_name = args.env_name + '-' + args.env_mode
args.env_name = args.env_name + '-EVAL' if args.eval_load else args.env_name

assert args.test_num % args.num_processes == 0
seed_interval = int(args.test_num/args.num_processes)
testSeed = config.envConfig.testSeed

gpu = "cuda:"+str(args.gpu)
device = torch.device(gpu if args.cuda else "cpu")
env = make_vec_envs(
    args.env_name,
    testSeed,
    args.num_processes,
    None,
    None,
    device=device,
    env_mode='test',
    seed_interval = seed_interval)

# We need to use the same statistics for normalization as used in training
if not args.eval_load:
    agent, ret_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))[0:2]
    actor_critic = agent.actor_critic
else:
    actor_critic,ret_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))[0:2]
actor_critic.to(device)

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ret_rms = ret_rms

#init
num_processes = args.num_processes 
test_reward_mat = [[] for _ in range(num_processes)]
episodes = 0
start_time = time.time()

recurrent_hidden_states = torch.zeros(num_processes,actor_critic.recurrent_hidden_state_size).to(device)
masks = torch.zeros(num_processes, 1, device=device)
obs = env.reset()
v_obs, *action_masks = env.get_action_mask()
while episodes < args.test_num:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, v_obs,recurrent_hidden_states, masks, action_masks, 
            deterministic=args.det)

    # Obser reward and next obs
    obs, _, done, infos = env.step(action)
    v_obs, *action_masks = env.get_action_mask()

    masks = torch.tensor(
        [[0.0] if done_ else [1.0] for done_ in done],
        dtype=torch.float32,
        device=device)
    
    for i in range(num_processes):
        info = infos[i]
        test_episode_rewards = test_reward_mat[i]
        if 'episode' in info.keys():
            this_grade = info['episode']['r']
            if len(test_episode_rewards) < seed_interval:
                test_episode_rewards.append(this_grade)
                episodes += 1
                
                print('Time {}: episode {}, this test grade = {:.3f}'
                      .format(time.strftime("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)),
                              episodes,this_grade))

log = ExcelLog(name = args.env_name + '-test',isLog=True)
result = []
for e in range(num_processes):
    for f in range(seed_interval):
        grade = test_reward_mat[e][f]
        log.saveTest(e*seed_interval+f+1,grade)
        result.append(grade)
print('Test set size:', len(result))
#print('Test set average grade:', )
print("Test set result: min: {}, average: {}, max:{}".format(
           min(result), sum(result)/len(result), max(result)))  
        

