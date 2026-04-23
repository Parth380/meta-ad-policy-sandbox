from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Advertiser CRM API")

ADVERTISERS = {
    "adv_001": {
        "name": "HealthPlus LLC",
        "prior_violations": 3,
        "account_age_days": 45,
        "summary": "3 prior violations for unverified health claims. High risk."
    },
    "adv_002": {
        "name": "QuickCash Corp",
        "prior_violations": 5,
        "account_age_days": 12,
        "summary": "5 prior violations: predatory lending. Account flagged."
    },
    "adv_003": {
        "name": "BrandSafe Inc",
        "prior_violations": 0,
        "account_age_days": 730,
        "summary": "No prior violations. Established advertiser."
    },
    "adv_004": {
        "name": "TrustMe Pharma",
        "prior_violations": 1,
        "account_age_days": 180,
        "summary": "1 prior violation: unapproved drug claim. Under review."
    },
    "adv_005": {
        "name": "YouthFinance App",
        "prior_violations": 2,
        "account_age_days": 30,
        "summary": "2 violations: targeting minors with financial products."
    }
}

@app.get("/advertiser/{advertiser_id}")
def get_advertiser(advertiser_id: str):
    if advertiser_id in ADVERTISERS:
        return ADVERTISERS[advertiser_id]
    return {
        "name": "Unknown Advertiser",
        "prior_violations": 0,
        "account_age_days": 7,
        "summary": "New unverified advertiser. No history. Treat with caution."
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "advertiser-crm"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)