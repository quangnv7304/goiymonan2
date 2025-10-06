"""CLI to run long SARSA training with resume/save options."""
import argparse
import os
import gym
from sarsa_agent import SARSAAgent
from sarsa_trainer import train_sarsa


def parse_args():
    p = argparse.ArgumentParser(description="Train SARSA agent on an environment")
    p.add_argument('--env', default='FrozenLake-v1', help='Gym environment id')
    p.add_argument('--episodes', type=int, default=10000)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--epsilon', type=float, default=0.2)
    p.add_argument('--epsilon-decay', type=float, default=0.9995)
    p.add_argument('--min-epsilon', type=float, default=0.01)
    p.add_argument('--save-path', default='sarsa_table.json')
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--render', action='store_true')
    p.add_argument('--no-decay', dest='decay', action='store_false')
    p.set_defaults(decay=True)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Training on env={args.env} episodes={args.episodes} max_steps={args.max_steps}")
    env = gym.make(args.env)
    actions = list(range(env.action_space.n))

    agent = SARSAAgent(actions, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, q_table_path=args.save_path)

    rewards = train_sarsa(env, agent, episodes=args.episodes, max_steps=args.max_steps,
                          decay_epsilon=args.decay, min_epsilon=args.min_epsilon, epsilon_decay=args.epsilon_decay,
                          save_every=args.save_every, render=args.render)
    print("Training finished. Saved Q-table to", args.save_path)


if __name__ == '__main__':
    main()
