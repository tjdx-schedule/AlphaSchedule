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

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device_cpu, 'train')
    net_dict = {
            # 'recurrent': args.recurrent_policy,
            # 'mlp_hiddens': [512,256],
            #'mix_hiddens': [256]
                }
    if not args.load:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            envs.value_observation_space.shape,
            base_kwargs=net_dict)
        log = ExcelLog(None,args,args.excel_save)
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            recurrent_accelerate=args.recurrent_accelerate)
    else:
        agent, ret_rms, log = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))#map_location=lambda storage, loc: storage
        actor_critic = agent.actor_critic
        log.setLog(args,args.excel_save)
        vec_norm = get_vec_normalize(envs)
        if vec_norm is not None:
            vec_norm.ret_rms = ret_rms
        print('load model:', args.env_name, 'success!')
    actor_critic.to(device_gpu)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space, envs.value_observation_space.shape,
                              actor_critic.recurrent_hidden_state_size, device_gpu)

    obs = envs.reset()
    v_obs, *action_masks = envs.get_action_mask()
    rollouts.init_zero_step(obs, v_obs, action_masks)
    
    deque_size = 100
    episode_rewards = deque(maxlen=deque_size)
    
    start_time = time.time()
    j = 0
    try:
        while True:
            j += 1
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    rollouts_obs, rollouts_v_obs,rollouts_recurrent_hidden_states, rollouts_masks, \
                         rollouts_action_masks = rollouts.getStep(step)
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts_obs, rollouts_v_obs, rollouts_recurrent_hidden_states,
                        rollouts_masks, rollouts_action_masks)
                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)
                v_obs, *action_masks = envs.get_action_mask()
    
                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
    
                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                rollouts.insert(obs, v_obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks, action_masks)
    
            with torch.no_grad():
                rollouts_obs, rollouts_v_obs, rollouts_recurrent_hidden_states, rollouts_masks, _ \
                    = rollouts.getStep(-1)
                next_value = actor_critic.get_value(
                    rollouts_obs, rollouts_v_obs, rollouts_recurrent_hidden_states,
                    rollouts_masks).detach()    
            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, False)
            value_loss, action_loss, dist_entropy, loss = agent.update(rollouts)
            rollouts.after_update()
    
            # save for every interval-th episode or for the last epoch
            if (j % args.save_interval == 0) and args.save_dir != "":
                save_model(args.save_dir,args.env_name,args.algo,
                           actor_critic,getattr(utils.get_vec_normalize(envs), 'ret_rms', None),
                           agent,log)
    
            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                print(
                    "Time {}: Updates {}, num timesteps {}, Last {} training episodes:\n\
entropy/value/policy/loss {:.3f}/{:.3f}/{:.3f}/{:.4f}, min/mean/max reward {:.3f}/{:.3f}/{:.3f}"
                    .format(time.strftime("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)),
                            j, total_num_steps,len(episode_rewards), 
                            dist_entropy, value_loss,action_loss,loss,
                            np.min(episode_rewards),np.mean(episode_rewards),np.max(episode_rewards)))
                if j % (args.log_interval * 3) == 0 and len(episode_rewards) >= deque_size:
                    log.saveTrain(np.mean(episode_rewards),time.strftime("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)))
                    log.saveLoss(dist_entropy, value_loss,action_loss,loss)
                            
            if (args.eval_interval is not None and len(episode_rewards) > 1
                    and j % args.eval_interval == 0):
                ret_rms = getattr(utils.get_vec_normalize(envs), 'ret_rms', None)
                evalMax,mean_eval = evaluate(actor_critic, ret_rms, args.env_name, None,
                                   args.num_processes, eval_log_dir, device_gpu, args.env_mode,
                                   log.getEval, args.algo, args.save_dir,args.eval_num)
                log.setEval(evalMax)
                log.saveTest(mean_eval)
            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                print("")
    except KeyboardInterrupt:
        save_model(args.save_dir,args.env_name,args.algo,
           actor_critic,getattr(utils.get_vec_normalize(envs), 'ret_rms', None),
           agent,log)


if __name__ == "__main__":
    main()
