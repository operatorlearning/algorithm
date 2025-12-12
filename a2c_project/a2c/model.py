import tensorflow as tf
import functools
from a2c.utils import Scheduler, find_trainable_variables


class Model:
    """A2C模型"""

    def __init__(self, policy, env, nsteps, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5, lr=7e-4, alpha=0.99, epsilon=1e-5,
                 total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf.get_default_session()
        nenvs = env.num_envs
        nbatch = nenvs * nsteps

        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            step_model = policy(sess, env.observation_space, env.action_space, nenvs, 1, reuse=False)
            train_model = policy(sess, env.observation_space, env.action_space, nbatch, nsteps, reuse=True)

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        vf_loss = tf.reduce_mean(tf.square(tf.squeeze(train_model.vf) - R))

        entropy = tf.reduce_mean(train_model.pd.entropy())

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("a2c_model")
        grads = tf.gradients(loss, params)

        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr_scheduler = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            cur_lr = lr_scheduler.value()

            td_map = {
                train_model.X: obs,
                A: actions,
                ADV: advs,
                R: rewards,
                LR: cur_lr
            }

            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train], td_map)

            return policy_loss, value_loss, policy_entropy

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state

        tf.global_variables_initializer().run(session=sess)
