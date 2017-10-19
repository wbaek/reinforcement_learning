import logging
import random
import gym
from core.environment import Env


def main(args):
    logging.info( args )

    env = gym.make(args.game)
    env = Env(env, resized_width=84, resized_height=84, agent_history_length=4)
    num_actions = len(env.gym_actions)
    state = env.get_initial_state()

    terminal = False
    while not terminal:
        action_index = random.choice(list(range(num_actions)))
        new_state, reward, terminal = env.step(action_index)
        env.env.render()

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game',    type=str,   default='Breakout-v0')
    parser.add_argument('--log-filename', type=str,   default='')
    args = parser.parse_args()

    if not args.log_filename:
        logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main( args )
