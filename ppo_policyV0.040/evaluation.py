import os

import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.game.envs import make_vec_envs


def evaluate(actor_critic, ret_rms, env_name, seed, 
             num_processes, eval_log_dir, device, env_mode,
             eval_max, algo, save_dir = "",eval_num = 80):
    seed = 0
    
    assert eval_num % num_processes == 0
    seed_interval = int(eval_num/num_processes)
    
    eval_reward_mat = [[] for _ in range(num_processes)]
    episodes = 0
    
    eval_envs = make_vec_envs(env_name, seed, num_processes, None, eval_log_dir, 
                              device, 'val', seed_interval=seed_interval)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ret_rms = ret_rms

    obs = eval_envs.reset()
    v_obs, *eval_action_masks = eval_envs.get_action_mask()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while episodes < eval_num:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                v_obs,
                eval_recurrent_hidden_states,
                eval_masks,
                eval_action_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)
        v_obs, *eval_action_masks = eval_envs.get_action_mask()

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for i in range(num_processes):
            info = infos[i]
            eval_episode_rewards = eval_reward_mat[i]
            if 'episode' in info.keys():
                if len(eval_episode_rewards) < seed_interval:
                    eval_episode_rewards.append(info['episode']['r'])
                    episodes += 1

    eval_envs.close()
    
    eval_reward_mat = np.array(eval_reward_mat)
    mean_eval = np.mean(eval_reward_mat) 
    print("Evaluation using {} episodes: mean reward {:.5f}".format(
        episodes, mean_eval))
    
    if mean_eval >= eval_max:
        eval_max = mean_eval
        save_model(save_dir,env_name,algo,actor_critic,ret_rms,mode = 'EVAL')
        
    return eval_max,mean_eval

def save_model(save_dir,name,algo,model,rms,agent=None,log=None,mode=''):
    if save_dir != "":
        save_path = os.path.join(save_dir,algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        if mode != '':
            name += '-' + mode
            
        if agent is not None:
            agent.actor_critic.save_weight(save_path,rms)
        else:
            model.save_weight(save_path,rms)
        torch.save([
            agent if agent is not None else model,
            rms,
            log
        ], os.path.join(save_path, name + ".pt"))