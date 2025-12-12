import time
import gym
import tensorflow as tf
from collections import deque
from a2c.model import Model
from a2c.runner import Runner
from a2c.policies import build_policy
from a2c.utils import explained_variance


def make_atari_env(env_id, num_env=8):
    """创建Atari环境（简化版）"""
    from gym.wrappers import TimeLimit

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env = TimeLimit(env, max_episode_steps=10000)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i) for i in range(num_env)])


class SubprocVecEnv:
    """简化的向量化环境"""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        env = self.envs[0]
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        return np.array([env.reset() for env in self.envs])

    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs, rews, dones, infos = zip(*results)

        for i, done in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()

        return np.array(obs), np.array(rews), np.array(dones), infos


def train_a2c(env_id='PongNoFrameskip-v4', num_timesteps=10000000,
              seed=0, nsteps=5, num_env=8):
    """训练A2C"""
    tf.reset_default_graph()

    with tf.Session() as sess:
        # 创建环境
        env = make_atari_env(env_id, num_env)

        # 构建策略
        policy = build_policy(env, 'cnn')

        # 创建模型
        model = Model(
            policy=policy,
            env=env,
            nsteps=nsteps,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            lr=7e-4,
            alpha=0.99,
            epsilon=1e-5,
            total_timesteps=num_timesteps,
            lrschedule='linear'
        )

        # 创建runner
        runner = Runner(env, model, nsteps=nsteps, gamma=0.99)
        epinfobuf = deque(maxlen=100)

        nbatch = num_env * nsteps
        tstart = time.time()

        for update in range(1, num_timesteps // nbatch + 1):
            obs, states, rewards, masks, actions, values, epinfos = runner.run()
            epinfobuf.extend(epinfos)

            policy_loss, value_loss, policy_entropy = model.train(
                obs, states, rewards, masks, actions, values)

            nseconds = time.time() - tstart
            fps = int((update * nbatch) / nseconds)

            if update % 100 == 0 or update == 1:
                ev = explained_variance(values, rewards)
                print(f"Update: {update}")
                print(f"Total timesteps: {update * nbatch}")
                print(f"FPS: {fps}")
                print(f"Policy entropy: {policy_entropy:.4f}")
                print(f"Value loss: {value_loss:.4f}")
                print(f"Explained variance: {ev:.4f}")

                if epinfobuf:
                    print(f"Episode reward mean: {np.mean([ep['r'] for ep in epinfobuf]):.2f}")
                    print(f"Episode length mean: {np.mean([ep['l'] for ep in epinfobuf]):.2f}")
                print("-" * 50)

        return model


if __name__ == "__main__":
    import numpy as np

    train_a2c(num_timesteps=1000000)
