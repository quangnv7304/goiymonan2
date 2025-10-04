# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from agent import OnlineLearningAgent # <-- Đổi sang OnlineLearningAgent
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Định nghĩa Models (giống backend gửi) ---
class State(BaseModel):
    avail: List[str]
    history: List[Any]
    context: Optional[Dict[str, Any]] = None

class PredictRequest(BaseModel):
    state: State
    k: int
    possible_actions: List[int]

class FeedbackPayload(BaseModel):
    state: State
    action: int # recipeId
    reward: float
    next_state: State
    done: bool

# --- Khởi tạo ---
app = FastAPI(title="Online Learning AI Service", version="2.1.0")
# Đổi tên file Q-table nếu muốn, ví dụ q_table_online.json
agent = OnlineLearningAgent(model_path="q_table_online.json") 

# --- API Endpoints ---
@app.post("/predict")
def predict(request: PredictRequest):
    # Endpoint này không thay đổi logic
    logger.info(f"Nhận request /predict: {request.dict()}")
    try:
        suggestion = agent.predict(request.state.dict(), request.possible_actions)
        logger.info(f"Trả về gợi ý: {suggestion}")
        return suggestion
    except Exception as e:
        logger.error(f"Lỗi trong quá trình predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/feedback")
def feedback(request: FeedbackPayload):
    """
    Nhận feedback và kích hoạt quá trình học online.
    """
    logger.info(f"Nhận được Feedback để học: {request.dict()}")
    try:
        agent.learn(
            state=request.state.dict(),
            action_id=request.action,
            reward=request.reward,
            next_state=request.next_state.dict(),
            done=request.done
        )
        return {"status": "ok", "message": "Agent has learned from feedback"}
    except Exception as e:
        logger.error(f"Lỗi trong quá trình learn: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ... endpoint "/" health_check giữ nguyên ...