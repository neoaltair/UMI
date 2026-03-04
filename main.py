import asyncio
import json
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd

from federated_core import run_federated_rounds, build_global_model

app = FastAPI(title="UMI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to store the latest training results
training_state = {
    "results": None,
    "is_training": False
}

class TrainConfig(BaseModel):
    n_rounds: int = 3
    mu: float = 0.01
    epsilon: float = 1.0

@app.get("/status")
async def get_status():
    return {"is_training": training_state["is_training"], "has_results": training_state["results"] is not None}

@app.post("/start-train")
async def start_train(config: TrainConfig):
    if training_state["is_training"]:
        return {"status": "error", "message": "Training already in progress"}
    
    # We'll run training in a background task and stream logs via SSE
    return {"status": "ok", "message": "Training started"}

@app.get("/stream-train")
async def stream_train(n_rounds: int = 3, mu: float = 0.01, epsilon: float = 1.0):
    async def event_generator():
        training_state["is_training"] = True
        
        queue = asyncio.Queue()

        def log_callback(event):
            asyncio.run_coroutine_threadsafe(queue.put(event), asyncio.get_event_loop())

        # Run training in a separate thread to avoid blocking the event loop
        def run_fl():
            try:
                res = run_federated_rounds(
                    n_rounds=n_rounds, 
                    mu=mu, 
                    epsilon=epsilon, 
                    verbose=False, 
                    log_callback=log_callback
                )
                training_state["results"] = res
                asyncio.run_coroutine_threadsafe(queue.put({"type": "complete", "data": "Training Finished"}), asyncio.get_event_loop())
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put({"type": "error", "message": str(e)}), asyncio.get_event_loop())
            finally:
                training_state["is_training"] = False

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, run_fl)

        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ["complete", "error"]:
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/history")
async def get_history():
    if not training_state["results"]:
        return {"status": "error", "message": "No training results available"}
    
    res = training_state["results"]
    return {
        "weight_history": res["weight_history"],
        "architectures": res["architectures"],
        "round_history": res["round_history"].to_dict(orient="records"),
        "feat_cols": res["feat_cols"]
    }

class PredictionInput(BaseModel):
    vitals: dict

@app.post("/predict")
async def predict(data: PredictionInput):
    if not training_state["results"]:
        return {"status": "error", "message": "Model not trained"}
    
    res = training_state["results"]
    gm = res["global_model"]
    scaler = res["scaler"]
    feat_cols = res["feat_cols"]
    
    vec = np.array([[data.vitals.get(f, 0) for f in feat_cols]])
    X_sc = scaler.transform(vec)
    prob = float(gm.predict_proba(X_sc)[0][1] * 100)
    
    # Analyze contribution
    contributions = {}
    local_weights = res["local_weights"]
    silo_sizes = res["silo_sizes"]
    total_samples = sum(silo_sizes.values())
    
    for hosp, (coef, _) in local_weights.items():
        # Contribution = (sample_size / total) * alignment_with_prediction
        # Simplified: weighted by sample size for now as requested
        contributions[hosp] = {
            "weight": silo_sizes[hosp] / total_samples,
            "sample_size": silo_sizes[hosp],
            "is_anomaly": False # This would be fetched from anomaly report in a real scenario
        }

    return {
        "probability": prob,
        "contributions": contributions,
        "feature_importance": dict(zip(feat_cols, gm.coef_[0].tolist()))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
