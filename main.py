# main.py  (Render - lightweight DB + API only)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from db import init_db, save_experiment, get_experiment  # your db.py

app = FastAPI(title="Activation Patching - Metadata API")

# allow requests from anywhere (Streamlit will server-side call it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

class ExperimentInput(BaseModel):
    prompt: str
    generated_text: str
    activation_traces: str
    explanation: str

@app.get("/")
async def root():
    return {"status": "ok", "message": "activation-patching metadata API"}

@app.post("/save")
async def save(payload: ExperimentInput):
    try:
        exp_id = save_experiment(
            payload.prompt,
            payload.generated_text,
            payload.activation_traces,
            payload.explanation,
        )
        return {"id": exp_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{id}")
async def results(id: int):
    row = get_experiment(id)
    if not row:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {
        "id": row[0],
        "prompt": row[1],
        "generated_text": row[2],
        "activation_traces": row[3],
        "explanation": row[4],
        "timestamp": row[5],
    }
