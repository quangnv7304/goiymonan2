import traceback
import gym
from sarsa_agent import SARSAAgent
from sarsa_trainer import train_sarsa

def main():
    print("Bắt đầu huấn luyện SARSA (run_train_once.py)")
    try:
        env = gym.make('FrozenLake-v1', is_slippery=False)
    except Exception as e:
        print("Không thể khởi tạo môi trường Gym FrozenLake:", e)
        return

    actions = list(range(env.action_space.n))
    agent = SARSAAgent(actions, alpha=0.5, gamma=0.99, epsilon=0.2, q_table_path='sarsa_table.json')
    try:
        rewards = train_sarsa(env, agent, episodes=200, max_steps=200,
                              decay_epsilon=True, min_epsilon=0.01, epsilon_decay=0.9995,
                              save_every=200, render=False)
        print("Huấn luyện hoàn tất. Tổng reward sample:", sum(rewards))
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    main()
