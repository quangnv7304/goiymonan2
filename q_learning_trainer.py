# q_learning_trainer.py
import requests
import random
import json
from collections import defaultdict
import time

# --- CẤU HÌNH ---
API_BASE_URL = "http://localhost:3000"
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
NUM_EPISODES = 100 # Tăng số lần huấn luyện để có model nền tốt hơn
MAX_STEPS_PER_EPISODE = 50
INGREDIENT_POOL = ["Trứng gà", "Dầu ăn", "Hành lá", "Thịt lợn", "Thịt bò", "Cà chua", "Hành tây", "Tỏi", "Gạo", "Mì", "Rau cải", "Nước mắm"]

class TrainingAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.action_space = self._get_action_space()
        print(f"Không gian hành động: {self.action_space}")

    def _get_action_space(self):
        try:
            response = requests.get(f"{API_BASE_URL}/recipes")
            response.raise_for_status()
            return [r['recipe_id'] for r in response.json()['recipes']]
        except Exception as e:
            print(f"Lỗi khi lấy action space: {e}")
            return []

    def get_state_hash(self, state_data):
        avail = tuple(sorted(state_data.get('available_ingredients', [])))
        context = state_data.get('context') or {}
        meal_time = context.get('meal_time', 'Unknown')
        history = context.get('history', [])
        last_3_actions = tuple(h.get('recipe_id') for h in history[-3:])
        return f"avail={avail}|meal_time={meal_time}|history={last_3_actions}"

    def choose_action(self, state_hash):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)
        else:
            q_values = self.q_table.get(state_hash, {})
            return max(q_values, key=q_values.get) if q_values else random.choice(self.action_space)

    def update_q_table(self, state_hash, action_id, reward, next_state_hash, done):
        old_value = self.q_table[state_hash][action_id]
        next_max_q = max(self.q_table[next_state_hash].values()) if not done and self.q_table[next_state_hash] else 0
        target = reward + self.gamma * next_max_q
        new_value = old_value + self.alpha * (target - old_value)
        self.q_table[state_hash][action_id] = new_value

    def update_epsilon(self):
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

def train():
    agent = TrainingAgent()
    if not agent.action_space: return

    for episode in range(NUM_EPISODES):
        initial_ingredients = random.sample(INGREDIENT_POOL, k=random.randint(3, 7))
        try:
            res = requests.post(f"{API_BASE_URL}/env/reset", json={"available_ingredients": initial_ingredients})
            res.raise_for_status()
            current_state_data = res.json()
        except Exception as e:
            print(f"Lỗi reset, bỏ qua episode: {e}")
            time.sleep(1)
            continue
        
        state_id = current_state_data['state_id']
        done = False
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            state_hash = agent.get_state_hash(current_state_data)
            action_id = agent.choose_action(state_hash)
            
            try:
                res = requests.post(f"{API_BASE_URL}/env/step", json={"state_id": state_id, "action_id": action_id})
                res.raise_for_status()
                step_data = res.json()
            except Exception as e:
                print(f"Lỗi step, kết thúc episode: {e}")
                break

            reward = step_data['reward']
            next_state_data = step_data['next_state']
            done = next_state_data['done']
            next_state_hash = agent.get_state_hash(next_state_data)
            
            agent.update_q_table(state_hash, action_id, reward, next_state_hash, done)

            current_state_data = next_state_data
            state_id = next_state_data['state_id']
            steps += 1
        
        agent.update_epsilon()
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{NUM_EPISODES}, Epsilon: {agent.epsilon:.4f}")

    print("\n--- Huấn luyện Hoàn tất ---")
    q_table_to_save = {state: {str(act): val for act, val in actions.items()} for state, actions in agent.q_table.items()}
    with open('q_table.json', 'w', encoding='utf-8') as f:
        json.dump(q_table_to_save, f, ensure_ascii=False, indent=4)
    print("Đã lưu Q-table vào file 'q_table.json'")

if __name__ == "__main__":
    train()