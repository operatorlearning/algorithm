import numpy as np
from a2c.utils import discount_with_dones


class Runner:
    """经验收集器"""

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.env = env
        self.model = model
        self.nsteps = nsteps
        self.gamma = gamma
        self.nenv = env.num_envs

        self.obs = env.reset()
        self.dones = [False for _ in range(self.nenv)]
        self.states = model.initial_state

    def run(self):
        """收集nsteps步的经验"""
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        epinfos = []

        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(
                self.obs, S=self.states, M=self.dones)

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            mb_rewards.append(rewards)

        mb_dones.append(self.dones)

        # 转换为numpy数组
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        # 调整维度
        mb_obs = mb_obs.swapaxes(0, 1).reshape(self.model.X.shape.as_list())
        mb_rewards = mb_rewards.swapaxes(0, 1)
        mb_actions = mb_actions.swapaxes(0, 1)
        mb_values = mb_values.swapaxes(0, 1)
        mb_dones = mb_dones.swapaxes(0, 1)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        # 计算折扣奖励
        if self.gamma > 0.0:
            last_values = self.model.value(self.obs, S=self.states, M=self.dones)
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)
                mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        mb_actions = mb_actions.flatten()

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, epinfos
