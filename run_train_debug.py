import traceback
import gym
import sarsa_trainer as t

env = gym.make('FrozenLake-v1', is_slippery=False)
actions = list(range(env.action_space.n))
agent = t.SARSAAgent(actions, alpha=0.5, gamma=0.99, epsilon=0.2, q_table_path='sarsa_table.json')
try:
    t.train_sarsa(env, agent, episodes=200, max_steps=200, decay_epsilon=True, min_epsilon=0.01, epsilon_decay=0.9995, save_every=200, render=False)
except Exception:
    traceback.print_exc()
print('done')
