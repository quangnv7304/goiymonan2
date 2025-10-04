# agent.py
import random
import json
import logging
from filelock import FileLock

logger = logging.getLogger(__name__)

class OnlineLearningAgent:
    def __init__(self, model_path: str, alpha=0.5, gamma=0.9, epsilon=0.1): # Tăng alpha lên 0.5
        self.model_path = model_path
        self.lock_path = model_path + ".lock"
        self.q_table = self._load_model()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        if self.q_table:
            logger.info(f"Agent đã tải thành công mô hình từ '{model_path}'")
        else:
            logger.warning(f"Không tìm thấy mô hình. Sẽ tạo mô hình mới khi có feedback.")

    def _load_model(self):
        try:
            with open(self.model_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_model(self):
        lock = FileLock(self.lock_path)
        try:
            with lock:
                with open(self.model_path, 'w', encoding='utf-8') as f:
                    json.dump(self.q_table, f, ensure_ascii=False, indent=4)
                logger.info(f"Đã cập nhật và lưu mô hình vào '{self.model_path}'")
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình: {e}")

    def _get_state_hash(self, state: dict) -> str:
        avail = tuple(sorted([i.lower() for i in state.get('avail', [])]))
        context = state.get('context') or {}
        meal_time = context.get('meal_time', 'Ăn trưa')
        history = state.get('history', [])
        # Sử dụng 3 recipeId cuối cùng trong history để tạo state hash
        last_3_actions = tuple(h.get('recipeId', h.get('recipe_id')) for h in history[-3:])
        return f"avail={avail}|meal_time={meal_time}|history={last_3_actions}"

    def predict(self, state: dict, possible_actions: list) -> dict:
        if not possible_actions:
            return {"chosen": None, "epsilon": self.epsilon, "message": "No possible actions provided."}

        state_hash = self._get_state_hash(state)
        logger.info(f"Xử lý cho state_hash: {state_hash}")

        # Epsilon-greedy: % nhỏ khám phá, còn lại là khai thác
        if random.uniform(0, 1) < self.epsilon:
            logger.info(f"Hành động Khám phá trong số {len(possible_actions)} actions.")
            chosen_id = random.choice(possible_actions)
            return {"chosen": chosen_id, "epsilon": self.epsilon, "message": "Exploration choice"}

        q_values_for_state = self.q_table.get(state_hash, {})
        if not q_values_for_state:
            logger.warning(f"Không có kinh nghiệm cho state_hash này. Fallback.")
            chosen_id = random.choice(possible_actions)
            return {"chosen": chosen_id, "epsilon": self.epsilon, "message": "Fallback: No experience"}
        
        valid_q_values = {int(act_id): score for act_id, score in q_values_for_state.items() if int(act_id) in possible_actions}
        if not valid_q_values:
            logger.warning(f"Không có Q-value hợp lệ. Fallback.")
            chosen_id = random.choice(possible_actions)
            return {"chosen": chosen_id, "epsilon": self.epsilon, "message": "Fallback: No valid Q-values"}

        best_action_id = max(valid_q_values, key=valid_q_values.get)
        logger.info(f"Hành động Khai thác. Gợi ý action_id: {best_action_id}")
        return {"chosen": best_action_id, "epsilon": self.epsilon, "message": "Exploitation choice"}

    def learn(self, state: dict, action_id: int, reward: float, next_state: dict, done: bool):
        state_hash = self._get_state_hash(state)
        next_state_hash = self._get_state_hash(next_state)
        action_id_str = str(action_id)

        logger.info(f"Bắt đầu học online: s={state_hash}, a={action_id}, r={reward}, s'={next_state_hash}")

        if state_hash not in self.q_table: self.q_table[state_hash] = {}
        if action_id_str not in self.q_table[state_hash]: self.q_table[state_hash][action_id_str] = 0.0

        old_value = self.q_table[state_hash][action_id_str]
        
        if reward < 0:
            new_value = -999.0
        else:
            # Nếu là like hoặc các reward khác, dùng công thức Q-learning như cũ
            next_max_q = 0.0
            if not done and next_state_hash in self.q_table and self.q_table[next_state_hash]:
                next_max_q = max(self.q_table[next_state_hash].values())

            target = reward + self.gamma * next_max_q
            new_value = old_value + self.alpha * (target - old_value)
        # =================================================================
        
        self.q_table[state_hash][action_id_str] = new_value
        logger.info(f"Đã cập nhật Q-value cho (s,a)=({state_hash},{action_id}) từ {old_value:.3f} -> {new_value:.3f}")

        self._save_model()