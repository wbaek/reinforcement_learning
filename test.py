import logging
import random
import gym
import numpy as np
import tensorflow as tf

from core.environment import Env
from core.worker import Network, Worker


def main(args):
    logging.info( args )

    env = gym.make(args.game)
    env = Env(env, resized_width=84, resized_height=84, agent_history_length=4)
    num_actions = len(env.gym_actions)

    global_net = Network(num_actions, -1, 'cpu')

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    rewards = []
    state = env.get_initial_state()

    terminal = False
    while not terminal:
        policy, value = global_net.predict(sess, state)
        action_index = np.argmax(policy) if args.best_policy else global_net.sampling(policy)
 
        state, reward, terminal = env.step(action_index)
        rewards.append( reward )
        if args.render:
            env.env.render()

    print('rewards:{}'.format( sum(rewards) ))

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game',    type=str,   default='Breakout-v0')
    parser.add_argument('--checkpoint',    type=str,   required=True, help='./checkpoints/recent.ckpt')
    parser.add_argument('--render',        action='store_true')
    parser.add_argument('--best-policy',   action='store_true')
    parser.add_argument('--log-filename',  type=str,   default='')
    args = parser.parse_args()

    if not args.log_filename:
        logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main( args )
