import time
import threading
import scipy.signal
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import logging
logger = logging.getLogger(__name__)

T = 200
EPSILON = 1e-10

class Network(object):
    def __init__(self, thread_id, output_actions_size, learning_rate=0.0001, beta=0.01):
        self.width, self.height, self.depth = 84, 84, 4
        self.thread_id = thread_id

        self.scope = 'net_' + str(thread_id)
        self.learning_rate = learning_rate
        self.beta = beta

        self.input_state = tf.placeholder("float", [None, self.height, self.width, self.depth])
        self.output_actions_size = output_actions_size

        self.advantage = tf.placeholder("float", [None])
        self.targets = tf.placeholder("float", [None])
        self.actions = tf.placeholder("float", [None, self.output_actions_size])

        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope(self.scope) as scope:
            #scope.reuse_variables()
            with slim.arg_scope([slim.layers.fully_connected], weights_regularizer=slim.l2_regularizer(1e-5)):
                l = slim.layers.conv2d(self.input_state, 16, [8, 8], stride=4, padding="VALID", scope='conv0')
                l = slim.layers.conv2d(l, 32, [4, 4], stride=2, padding="VALID", scope='conv1')
                l = slim.layers.flatten(l)
                l = slim.layers.fully_connected(l, 256, activation_fn=tf.nn.relu, scope='fc1')

                policy_logit = slim.layers.fully_connected(l, self.output_actions_size, activation_fn=None, scope='fc_policy')
                value_logit = slim.layers.fully_connected(l, 1, activation_fn=None, scope='fc_value')

        self.policy = tf.nn.softmax(policy_logit)
        self.value = value_logit

        # advantage loss
        log_pol = tf.log(self.policy + EPSILON)
        entropy_logit = -tf.reduce_sum(tf.multiply(log_pol, self.policy), reduction_indices=1)
        entropy_target = tf.reduce_sum(tf.multiply(log_pol, self.actions), reduction_indices=1)
        policy_loss = -tf.reduce_sum(entropy_target * self.advantage + entropy_logit * self.beta)
        
        value_loss = tf.nn.l2_loss(self.targets - self.value)

        self.loss = policy_loss + 0.5 * value_loss

        self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

    def tie_global_net(self, global_net):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads = tf.gradients(self.loss, self.train_vars)
        
        grads_and_vars = list(zip(self.grads, global_net.train_vars))
        self.opt = tf.group(self.optimizer.apply_gradients(grads_and_vars), tf.shape(self.input_state)[0])

        sync = [self.train_vars[j].assign(global_net.train_vars[j]) for j in range(len(self.train_vars))]
        self.sync_op = tf.group(*sync)

    def sync(self, session):
        session.run(self.sync_op)

    def predict(self, session, state):
        policy, value = session.run([self.policy, self.value], feed_dict={self.input_state: [state]})
        return policy[0], value[0][0]

    def sampling(self, probs):
        probs = probs - np.finfo(np.float32).epsneg
        histogram = np.random.multinomial(1, probs)
        action_index = int(np.nonzero(histogram)[0])
        return action_index
       
    def optimize(self, session, states, actions, targets, advantage):
        _, cost = session.run( [self.opt, self.loss], feed_dict={self.input_state: states,
                                                                   self.actions: actions,
                                                                   self.targets: targets,
                                                                   self.advantage: advantage})
        return cost

    
class Worker(threading.Thread):
    def __init__(self, thread_id, env, net, session, saver, checkpoint_dir, checkpoint_interval=1000000, gamma=0.99, tmax=5):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.env = env
        self.net = net

        self.session = session
        self.saver = saver
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval

        self.gamma = gamma
        self.tmax = tmax
        self.TMAX = 80000000

    def run(self):
        time.sleep(3 * self.thread_id)

        state = self.env.get_initial_state()

        global T
        counter = 0
        total_cost, episode_reward = 0, 0
        while T < self.TMAX:
            states, actions, prev_reward, state_values = [], [], [], []
            terminal = False

            t = 0
            self.net.sync(self.session)
            while not (terminal or (t >= self.tmax)):
                policy, value = self.net.predict(self.session, state)

                action_list = np.zeros([self.net.output_actions_size])
                action_index = self.net.sampling( policy )

                action_list[action_index] = 1

                actions.append(action_list)
                states.append(state)
                state_values.append(value)

                state, reward, terminal = self.env.step(action_index)
                clipped_reward = np.clip(reward, -1, 1)
                prev_reward.append(clipped_reward)

                episode_reward += reward

                counter += 1
                t += 1
                T += 1
                if T % self.checkpoint_interval < 200:
                    T += 200
                    self.saver.save(self.session, self.checkpoint_dir+"/currnet.ckpt" , global_step=T)
                
            R = 0.0 if terminal else self.net.predict(self.session, state)[1]
            state_values.append(R)
            targets = np.zeros(t)

            for i in range(t - 1, -1, -1):
                R = prev_reward[i] + self.gamma * R
                targets[i] = R

            delta = np.array(prev_reward) + self.gamma * np.array(state_values[1:]) - np.array(state_values[:-1])
            advantage = scipy.signal.lfilter([1], [1, -self.gamma], delta[::-1], axis=0)[::-1]
            
            total_cost += self.net.optimize(self.session, states, actions, targets, advantage)
            
            if terminal:
                state = self.env.get_initial_state()
                logger.info('THREAD:%02d TIME:%08d REWARD:%04d, COST:%.4f'%(self.thread_id, T, episode_reward, total_cost/counter))
                counter, total_cost, episode_reward = 0, 0, 0                



