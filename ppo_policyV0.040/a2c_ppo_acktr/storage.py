import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 v_obs_shape, recurrent_hidden_state_size, device):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.v_obs = torch.zeros(num_steps + 1, num_processes, *v_obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        
        act_num = [i.n for i in action_space]
        self.action_shape = len(act_num)
        # action_shape = len(action_space)
        self.actions = torch.zeros(num_steps, num_processes, self.action_shape)
        self.actions = self.actions.long()
        
        # act_num = [i.n for i in action_space]
        self.action_masks =[]
        for i in range(len(act_num)):
            if i == 0:
                action_mask = torch.ones(num_steps + 1, num_processes, act_num[0])
            elif i == 1:
                action_mask = torch.ones(num_steps + 1, num_processes, act_num[0], act_num[1])
            else:
                action_mask = torch.ones(num_steps + 1, num_processes, act_num[1], act_num[i])
            self.action_masks.append(action_mask)

        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0
        self.device = device

    def init_zero_step(self, obs, v_obs, action_masks):# stock_action_masks, \
                       # mach_action_masks, tool_action_masks, worker_action_masks):
        self.obs[0].copy_(obs)  
        self.v_obs[0].copy_(v_obs)
        
        for i in range(self.action_shape):
            self.action_masks[i].copy_(action_masks[i])
       
    def getStep(self,step):
        device = self.device
        
        obs = self.obs[step].to(device)
        v_obs = self.v_obs[step].to(device)
        recurrent_hidden_states = self.recurrent_hidden_states[step].to(device)
        masks = self.masks[step].to(device)
        
        action_masks = [] 
        for i in range(self.action_shape):
            action_masks.append(self.action_masks[i][step].to(device))

        return obs, v_obs, recurrent_hidden_states, masks, action_masks
                
    def insert(self, obs, v_obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, action_masks):
               # tool_action_masks, worker_action_masks):
        self.obs[self.step + 1].copy_(obs)
        self.v_obs[self.step + 1].copy_(v_obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        
        for i in range(self.action_shape):
            self.action_masks[i][self.step + 1].copy_(action_masks[i])

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.v_obs[0].copy_(self.v_obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        
        for i in range(self.action_shape):
            self.action_masks[i][0].copy_(self.action_masks[i][-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=False):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices].to(self.device)
            v_obs_batch = self.v_obs[:-1].view(-1, *self.v_obs.size()[2:])[indices].to(self.device)
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices].to(self.device)
            actions_batch = self.actions.view(-1,self.actions.size(-1))[indices].to(self.device)
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices].to(self.device)
            return_batch = self.returns[:-1].view(-1, 1)[indices].to(self.device)
            masks_batch = self.masks[:-1].view(-1, 1)[indices].to(self.device)
            old_action_log_probs_batch = self.action_log_probs.view(-1,1)[indices].to(self.device)
            
            action_mask_batch = []
            for i in range(self.action_shape):
                batch = self.action_masks[i][:-1].\
                    view(-1, *self.action_masks[i].size()[2:])[indices].to(self.device)
                action_mask_batch.append(batch)

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices].to(self.device)

            yield obs_batch, v_obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ, action_mask_batch


    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            v_obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            
            action_mask_batchs = [[] for i in range(self.action_shape)]
           
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                v_obs_batch.append(self.v_obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])             
                for i in range(self.action_shape):
                    action_mask_batchs[i].append(self.action_masks[i][:-1, ind])    

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            v_obs_batch = torch.stack(v_obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)            
            for i in range(self.action_shape):
                action_mask_batchs[i] = torch.stack(action_mask_batchs[i],1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1).to(self.device)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch).to(self.device)
            v_obs_batch = _flatten_helper(T, N, v_obs_batch).to(self.device)
            actions_batch = _flatten_helper(T, N, actions_batch).to(self.device)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch).to(self.device)
            return_batch = _flatten_helper(T, N, return_batch).to(self.device)
            masks_batch = _flatten_helper(T, N, masks_batch).to(self.device)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch).to(self.device)
            adv_targ = _flatten_helper(T, N, adv_targ).to(self.device)
            for i in range(self.action_shape):
                action_mask_batchs[i] = _flatten_helper(T, N, action_mask_batchs[i]).to(self.device)

            yield obs_batch, v_obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ, action_mask_batchs#stock_action_mask_batch, mach_action_mask_batch, tool_action_mask_batch, \
                #worker_action_mask_batch
