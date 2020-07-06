import argparse

from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq import DQN, wrap_atari_dqn

def main():
    """
    Run a trained model in an Atari environment.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--no-render', default=False, action="store_true", help="Disable rendering")

    args = parser.parse_args()
    env = make_atari(args.env)
    env = wrap_atari_dqn(env)
    model = DQN.load("model.zip", env)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            if not args.no_render:
                env.render()
            action, _ = model.predict(obs)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)
        # No render is only used for automatic testing
        if args.no_render:
            break


if __name__ == '__main__':
    main()
