import json
import random
import os
from collections import defaultdict

# expose public API from this module
__all__ = ["SARSAAgent", "OnlineLearningAgent"]

class SARSAAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1, q_table_path="sarsa_table.json"):
        """
        actions: list-like các action hợp lệ (ví dụ [0,1,2,3])
        alpha: learning rate
        gamma: discount factor
        epsilon: epsilon cho epsilon-greedy
        q_table_path: file lưu Q-table JSON
        """
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_path = q_table_path
        # SARSA table (bảng giá trị) là dict: { state_str: {action: value, ...}, ... }
        self.q = defaultdict(lambda: {a: 0.0 for a in self.actions})
        # nếu file tồn tại -> load
        if os.path.exists(self.q_table_path):
            try:
                self.load(self.q_table_path)
                print(f"Loaded SARSA table from {self.q_table_path}")
            except Exception as e:
                print("Không thể load SARSA table:", e)
        else:
            # backward compatibility: nếu còn file q_table.json cũ -> load và migrate
            legacy = "q_table.json"
            if os.path.exists(legacy):
                try:
                    self.load(legacy)
                    # ngay lập tức lưu sang tên mới
                    self.save(self.q_table_path)
                    print(f"Migrated legacy '{legacy}' -> '{self.q_table_path}'")
                except Exception as e:
                    print("Không thể migrate legacy Q-table:", e)

    def state_to_key(self, state):
        """
        Chuyển state thành key (string) để lưu Q-table.
        Nếu state đã là str, trả lại trực tiếp.
        Nếu state là tuple/list, chuyển thành str.
        Điều chỉnh nếu state trong repo của bạn có dạng khác.
        """
        if isinstance(state, str):
            return state
        try:
            return json.dumps(state, sort_keys=True)
        except:
            return str(state)

    def choose_action(self, state):
        """Epsilon-greedy: chọn action theo Q hiện tại"""
        key = self.state_to_key(state)
        # khám phá
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # khai thác: chọn action có Q lớn nhất (break ties ngẫu nhiên)
        q_vals = self.q[key]
        # ensure we choose among self.actions so returned action has the expected type
        values = [q_vals.get(a, q_vals.get(str(a), 0.0)) for a in self.actions]
        max_q = max(values) if values else 0.0
        max_actions = [a for a, v in zip(self.actions, values) if v == max_q]
        if not max_actions:
            # fallback: use keys from q_vals
            max_q2 = max(q_vals.values()) if q_vals else 0.0
            max_actions2 = [a for a, v in q_vals.items() if v == max_q2]
            return random.choice(max_actions2)
        return random.choice(max_actions)

    def update(self, state, action, reward, next_state, next_action, done):
        """
        Cập nhật Q theo công thức SARSA:
        Q(s,a) += alpha * (r + gamma * Q(s',a')*(1-done) - Q(s,a))
        Lưu ý: nếu done thì không thêm gamma * Q(s',a')
        """
        s_key = self.state_to_key(state)
        s_next_key = self.state_to_key(next_state)
        # ensure entries exist so .get won't fail and types are consistent
        _ = self.q[s_key]
        _ = self.q[s_next_key]
        q_sa = self.q[s_key].get(action, 0.0)
        q_snext_anext = 0.0
        if not done:
            q_snext_anext = self.q[s_next_key].get(next_action, 0.0)
        td_target = reward + (0 if done else self.gamma * q_snext_anext)
        td_error = td_target - q_sa
        new_q = q_sa + self.alpha * td_error
        self.q[s_key][action] = new_q
        return td_error

    def save(self, path=None):
        path = path or self.q_table_path
        # Ensure parent directory exists (if any)
        dirname = os.path.dirname(path)
        if dirname:
            try:
                os.makedirs(dirname, exist_ok=True)
            except Exception:
                # ignore directory creation errors, will surface on file open if real problem
                pass
        # chuyển defaultdict -> dict để dump JSON
        q_dump = {k: v for k, v in self.q.items()}
        # JSON không chấp nhận keys không là str; ở đây keys đã là str nhờ state_to_key
        with open(path, "w") as f:
            json.dump(q_dump, f, indent=2)
        print(f"SARSA table saved to {path}")

    def load(self, path=None):
        path = path or self.q_table_path
        # robustly handle malformed JSON / IO errors
        try:
            with open(path, "r") as f:
                q_loaded = json.load(f)
        except (ValueError, json.JSONDecodeError) as e:
            # corrupted file -> start with empty table and warn
            print(f"Warning: failed to parse SARSA table '{path}': {e}. Starting with empty table.")
            q_loaded = {}
        except Exception as e:
            # other IO errors -> re-raise so caller can handle if desired
            raise

        # convert back to defaultdict
        self.q = defaultdict(lambda: {a: 0.0 for a in self.actions})
        for k, v in q_loaded.items():
            # ensure actions set exist (nếu thiếu action trong file, gán 0)
            entry = {a: float(v.get(str(a), v.get(a, 0.0))) for a in self.actions}
            # also keep any numeric-string keys from older format
            for ak, av in v.items():
                try:
                    # try preserve non-listed action keys if any
                    entry[ak] = float(av)
                except:
                    pass
            self.q[k] = entry

    def set_epsilon(self, eps):
        self.epsilon = eps

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_gamma(self, gamma):
        self.gamma = gamma


class OnlineLearningAgent:
    """
    Adapter to expose a simple online API (predict, learn) that the server expects.
    Wraps the SARSAAgent but accepts possible_actions at predict time and
    performs an on-policy SARSA update when `learn` is called.
    """
    def __init__(self, model_path="sarsa_table.json", alpha=0.1, gamma=0.99, epsilon=0.1):
        # initialize SARSAAgent with an empty action list; actions are managed per-state
        self.sarsa = SARSAAgent(actions=[], alpha=alpha, gamma=gamma, epsilon=epsilon, q_table_path=model_path)

    def _ensure_actions_for_state(self, state, actions):
        key = self.sarsa.state_to_key(state)
        for a in actions:
            # keep action keys as the same type used by callers (usually int)
            if a not in self.sarsa.q[key]:
                self.sarsa.q[key][a] = 0.0
            # also ensure the global actions list contains this action so choose_action uses it
            if a not in self.sarsa.actions:
                self.sarsa.actions.append(a)

    def predict(self, state: dict, possible_actions: list):
        """Return a suggestion for an action given the current state and list of possible actions.
        Returns a dict with selected action and current q-values for the provided actions.
        """
        # ensure q entries exist for the provided actions
        self._ensure_actions_for_state(state, possible_actions)
        # temporarily set the agent's action list so choose_action explores among possible_actions
        prev_actions = list(self.sarsa.actions)
        try:
            self.sarsa.actions = list(possible_actions)
            action = self.sarsa.choose_action(state)
        finally:
            self.sarsa.actions = prev_actions

        key = self.sarsa.state_to_key(state)
        # prepare q-values in a JSON-serializable way (cast keys to str)
        qvals = {str(k): float(v) for k, v in self.sarsa.q[key].items()}
        return {"action": action, "q_values": qvals}

    def learn(self, state: dict, action_id, reward: float, next_state: dict, done: bool):
        """
        Perform an on-policy SARSA update using the provided transition.

        Since the server does not provide the next action, we pick the greedy next action
        according to the current Q (on-policy approximation). If no next actions exist,
        we use the current action as next_action (bootstrapping to itself).
        """
        # Make sure current state/action entries exist
        s_key = self.sarsa.state_to_key(state)
        ns_key = self.sarsa.state_to_key(next_state)
        # ensure dictionaries exist
        _ = self.sarsa.q[s_key]
        _ = self.sarsa.q[ns_key]

        # determine next_action: pick argmax over q-values in next_state if available
        q_next = self.sarsa.q[ns_key]
        next_action = action_id
        if q_next:
            # Prefer choosing next_action from the agent's known action list to keep types consistent
            try:
                best = None
                best_val = None
                for a in self.sarsa.actions:
                    val = q_next.get(a, q_next.get(str(a), 0.0))
                    if best is None or float(val) > best_val:
                        best = a
                        best_val = float(val)
                if best is not None:
                    next_action = best
            except Exception:
                # fallback to previous behavior
                try:
                    next_action = max(q_next.items(), key=lambda kv: float(kv[1]))[0]
                except Exception:
                    next_action = action_id

        # perform SARSA update
        try:
            td_error = self.sarsa.update(state, action_id, reward, next_state, next_action, done)
        except Exception as e:
            # make a best-effort: if update fails because action keys are strings/numbers, coerce keys
            # convert existing q entries to use same action key types
            # fallback: ensure numeric key exists
            if action_id not in self.sarsa.q[s_key]:
                self.sarsa.q[s_key][action_id] = 0.0
            if next_action not in self.sarsa.q[ns_key]:
                self.sarsa.q[ns_key][next_action] = 0.0
            td_error = self.sarsa.update(state, action_id, reward, next_state, next_action, done)

        # persist the SARSA table after learning (simple, safe default)
        try:
            self.sarsa.save()
        except Exception:
            # ignore save errors for now; server will log exceptions
            pass

        return {"status": "ok", "td_error": td_error}

    def save(self, path=None):
        self.sarsa.save(path)

    def set_epsilon(self, eps: float):
        self.sarsa.set_epsilon(eps)

    def set_alpha(self, a: float):
        self.sarsa.set_alpha(a)

    def set_gamma(self, g: float):
        self.sarsa.set_gamma(g)
