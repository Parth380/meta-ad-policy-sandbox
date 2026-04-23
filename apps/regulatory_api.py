from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Regulatory DB API")

REGULATIONS = {
    "healthcare": {
        "policy_summary": "Claims require FDA approval. No 'guaranteed results' allowed.",
        "risk_level": "high"
    },
    "financial": {
        "policy_summary": "Requires SEC registration. Prohibited: predatory APR > 36%.",
        "risk_level": "high"
    },
    "general": {
        "policy_summary": "Standard standards apply. No deceptive claims.",
        "risk_level": "low"
    }
}

@app.get("/regulations/{category}")
def get_regulations(category: str):
    return REGULATIONS.get(category.lower(), REGULATIONS["general"])

@app.get("/health")
def health():
    return {"status": "ok", "service": "regulatory-db"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)