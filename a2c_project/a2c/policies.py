import tensorflow as tf
import numpy as np
from a2c.utils import conv, fc, conv_to_fc, ortho_init


class CnnPolicy:
    """CNN策略网络"""

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        self.sess = sess
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n

        X = tf.placeholder(tf.uint8, ob_shape)

        with tf.variable_scope("model", reuse=reuse):
            h = tf.cast(X, tf.float32) / 255.
            h = tf.nn.relu(conv(h, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
            h = tf.nn.relu(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
            h = tf.nn.relu(conv(h, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
            h = conv_to_fc(h)
            h = tf.nn.relu(fc(h, 'fc1', nh=512, init_scale=np.sqrt(2)))

            pi = fc(h, 'pi', nact, init_scale=0.01)
            vf = fc(h, 'v', 1)

        self.pd = CategoricalPd(pi)
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)

        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v[:, 0], self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})[:, 0]

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.action = a0


class CategoricalPd:
    """分类概率分布"""

    def __init__(self, logits):
        self.logits = logits

    def sample(self):
        noise = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(noise)), axis=-1)

    def neglogp(self, x):
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=one_hot_actions)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)


def build_policy(env, policy_network='cnn'):
    """构建策略"""
    if policy_network == 'cnn':
        return CnnPolicy
    else:
        raise ValueError(f"Unknown policy network: {policy_network}")
