# q_learning_trainer.py
import time
import numpy as np
from sarsa_agent import SARSAAgent


def _unpack_step(env, action):
    """Bảo đảm tương thích env.step trả 3 hoặc 4 giá trị."""
    res = env.step(action)
    # gymnasium: (obs, reward, terminated, truncated, info)
    if isinstance(res, tuple):
        if len(res) == 5:
            next_state, reward, terminated, truncated, info = res
            done = bool(terminated or truncated)
            return next_state, reward, done, info
        try:
            # gym classic: (next_state, reward, done, info)
            next_state, reward, done, info = res
            return next_state, reward, done, info
        except ValueError:
            # fallback: (next_state, reward, done)
            next_state, reward, done = res
            info = {}
            return next_state, reward, done, info
    # unknown format -> raise
    raise ValueError(f"Unexpected env.step return type: {type(res)} / {res}")


def _reset_env(env):
    """Xử lý cả Gym và Gymnasium: reset có thể trả obs hoặc (obs, info)."""
    res = env.reset()
    try:
        # gymnasium: (obs, info)
        obs, info = res
        return obs
    except Exception:
        # gym: obs trực tiếp
        return res


def train_sarsa(env, agent: SARSAAgent, episodes=1000, max_steps=200,
                decay_epsilon=True, min_epsilon=0.01, epsilon_decay=0.995,
                save_every=100, render=False):
    rewards_per_ep = []
    for ep in range(1, episodes + 1):
        state = _reset_env(env)
        action = agent.choose_action(state)  # chọn a cho s (on-policy)
        total_reward = 0.0

        for t in range(max_steps):
            if render:
                try:
                    env.render()
                except:
                    pass
            next_state, reward, done, info = _unpack_step(env, action)
            next_action = agent.choose_action(next_state)  # chọn a' (on-policy)
            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            total_reward += reward
            if done:
                break

        # epsilon decay
        if decay_epsilon:
            agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)

        rewards_per_ep.append(total_reward)

        if save_every and ep % save_every == 0:
            agent.save()

        if ep % max(1, episodes // 10) == 0:
            avg_recent = np.mean(rewards_per_ep[-max(1, episodes // 20):])
            print(f"[Train] Ep {ep}/{episodes} | reward {total_reward:.3f} | avg_recent {avg_recent:.3f} | eps {agent.epsilon:.4f}")

    agent.save()
    return rewards_per_ep

if __name__ == "__main__":
    # ví dụ thử nhanh với OpenAI Gym (nếu bạn cài gym)
    try:
        import gym
        env = gym.make("FrozenLake-v1", is_slippery=False)
        actions = list(range(env.action_space.n))
        agent = SARSAAgent(actions, alpha=0.5, gamma=0.99, epsilon=0.2, q_table_path="sarsa_table.json")
        rewards = train_sarsa(env, agent, episodes=2000, max_steps=200,
                              decay_epsilon=True, min_epsilon=0.01, epsilon_decay=0.9995,
                              save_every=200, render=False)
    except Exception as e:
        print("[Example] Cần gym để chạy ví dụ. Lỗi:", e)
        print("[Example] Bạn có thể gọi train_sarsa(env, agent, ...) trực tiếp trong mã của bạn.")
