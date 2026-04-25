from fastapi import FastAPI
import uvicorn
import random

app = FastAPI(title="Regulatory DB API")

REGULATIONS = {
    "healthcare": {
        "policy_hint": "Health claims require FDA verification. No unverified cures or prescription bypasses.",
        "violations": [
            {"type": "unverified_cure_claim", "confidence": 0.9},
            {"type": "prescription_bypass", "confidence": 0.4}
        ]
    },
    "financial": {
        "policy_hint": "SEC registration required. No guaranteed returns or predatory lending.",
        "violations": [
            {"type": "guaranteed_returns", "confidence": 0.85},
            {"type": "predatory_lending", "confidence": 0.5}
        ]
    },
    "targeting": {
        "policy_hint": "Age-restricted products cannot target minors. age_min must be >= 18.",
        "violations": [
            {"type": "minor_targeting", "confidence": 0.95}
        ]
    },
    "ambiguous": {
        "policy_hint": "Policy applicability is unclear. Gather additional signals before deciding.",
        "violations": [
            {"type": "possible_misleading_claim", "confidence": 0.45},
            {"type": "unverified_endorsement", "confidence": 0.5}
        ]
    },
    "general": {
        "policy_hint": "Standard advertising standards apply. No deceptive claims.",
        "violations": []
    },
    "none": {
        "policy_hint": "Standard advertising standards apply. No deceptive claims.",
        "violations": []
    }
}

@app.get("/regulations/{category}")
def get_regulations(category: str):
    if random.random() < 0.1:
        return {"error": "service_unavailable", "retryable": True}
    return REGULATIONS.get(category.lower(), REGULATIONS["general"])

@app.get("/health")
def health():
    return {"status": "ok", "service": "regulatory-db"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)