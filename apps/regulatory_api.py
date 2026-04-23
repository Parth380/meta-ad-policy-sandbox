# regulatory_api.py
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Regulatory DB API")

REGULATIONS = {
    "healthcare": {
        "policy_summary": (
            "Health claims require FDA approval. "
            "Prohibited: unverified cure claims, 'guaranteed results', "
            "prescription drug sales without authorization."
        ),
        "risk_level": "high"
    },
    "financial": {
        "policy_summary": (
            "Financial ads require SEC registration. "
            "Prohibited: guaranteed returns, predatory APR above 36%, "
            "high-pressure investment tactics."
        ),
        "risk_level": "high"
    },
    "targeting": {
        "policy_summary": (
            "Age-restricted products cannot target minors. "
            "Financial and healthcare products require age_min >= 18."
        ),
        "risk_level": "high"
    },
    "general": {
        "policy_summary": "Standard advertising standards apply. No deceptive claims.",
        "risk_level": "low"
    },
    "none": {
        "policy_summary": "Standard advertising standards apply. No deceptive claims.",
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