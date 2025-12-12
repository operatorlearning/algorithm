import os
import numpy as np
import tensorflow as tf
from collections import deque

def discount_with_dones(rewards, dones, gamma):
    """计算折扣奖励"""
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)
        discounted.append(r)
    return discounted[::-1]

def find_trainable_variables(key):
    """查找可训练变量"""
    return tf.trainable_variables(key)

def make_path(f):
    """创建目录"""
    return os.makedirs(f, exist_ok=True)

def ortho_init(scale=1.0):
    """正交初始化"""
    def _ortho_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def conv(x, scope, nf, rf, stride, pad='VALID', init_scale=1.0):
    """卷积层"""
    channel_ax = 3
    strides = [1, stride, stride, 1]
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nin, nf]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [1, 1, 1, nf], initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d(x, w, strides=strides, padding=pad) + b

def fc(x, scope, nh, init_scale=1.0):
    """全连接层"""
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b

def conv_to_fc(x):
    """卷积输出转全连接"""
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x

def explained_variance(ypred, y):
    """解释方差"""
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary

class Scheduler(object):
    """学习率调度器"""
    def __init__(self, v, nvalues, schedule='linear'):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues
        self.schedule = schedule

    def value(self):
        if self.schedule == 'constant':
            current_value = self.v
        elif self.schedule == 'linear':
            current_value = self.v * (1 - self.n / self.nvalues)
        else:
            current_value = self.v
        self.n += 1.
        return current_value
