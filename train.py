import os
import logging
import random
import gym
import tensorflow as tf
from tensorflow.python.client import device_lib

from core.environment import Env
from core.worker import Network, Worker


def main(args):
    logging.info( args )
    device = 'gpu' if args.gpu else 'cpu'

    devices = device_lib.list_local_devices()
    num_gpus = len([d for d in devices if '/gpu' in d.name])
 
    env = gym.make(args.game)
    env = Env(env, resized_width=84, resized_height=84, agent_history_length=4)
    num_actions = len(env.gym_actions)

    global_net = Network(num_actions, -1, 'cpu')
    actor_networks = []
    for t in range(args.threads):
        device_index = 0 if device is 'cpu' else (t if args.threads <= num_gpus else 0)
        n = Network(num_actions, t, device, device_index)
        n.tie_global_net(global_net)
        actor_networks.append(n)

    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=args.threads, inter_op_parallelism_threads=args.threads))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    threads = []
    for t, net in enumerate(actor_networks):
        e = Env(gym.make(args.game), net.width, net.height, net.depth)
        w = Worker(t, e, net, sess, saver, args.checkpoint_dir)
        w.start()
        threads.append(w)

    for t in threads:
        t.join()
    

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game',     type=str,   default='Breakout-v0')
    parser.add_argument('-t', '--threads',  type=int,   default=5)
    parser.add_argument('--checkpoint-dir', type=str,   default='./checkpoints/')
    parser.add_argument('--gpu',            action="store_true")
    parser.add_argument('--log-filename',   type=str,   default='')
    args = parser.parse_args()

    if not args.log_filename:
        logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main( args )
