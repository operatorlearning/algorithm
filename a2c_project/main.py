import gym
import numpy as np
import tensorflow as tf
from a2c.model import Model
from a2c.runner import Runner
from a2c.policies import CnnPolicy


class SimpleVecEnv:
    """简单的向量化环境包装器"""

    def __init__(self, env_id='CartPole-v1', n_envs=4):
        self.envs = [gym.make(env_id) for _ in range(n_envs)]
        self.num_envs = n_envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        return np.array([env.reset() for env in self.envs])

    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs, rewards, dones, infos = zip(*results)

        for i, done in enumerate(dones):
            if done:
                obs = list(obs)
                obs[i] = self.envs[i].reset()

        return np.array(obs), np.array(rewards), np.array(dones), list(infos)


def main():
    """主函数 - 训练CartPole"""
    print("开始训练A2C on CartPole-v1...")

    tf.reset_default_graph()

    with tf.Session() as sess:
        env = SimpleVecEnv('CartPole-v1', n_envs=4)

        # 使用简单的MLP策略
        from a2c.policies import build_policy
        policy_fn = build_policy(env, 'cnn')

        model = Model(
            policy=policy_fn,
            env=env,
            nsteps=5,
            total_timesteps=100000
        )

        runner = Runner(env, model, nsteps=5)

        for update in range(1, 1001):
            obs, states, rewards, masks, actions, values, epinfos = runner.run()

            policy_loss, value_loss, policy_entropy = model.train(
                obs, states, rewards, masks, actions, values)

            if update % 100 == 0:
                print(f"Update {update}: reward_mean={np.mean(rewards):.2f}")

        print("训练完成!")


if __name__ == "__main__":
    main()
