from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Advertiser CRM API")

ADVERTISERS = {
    "adv_001": {"prior_violations": 3, "summary": "High-risk repeat offender."},
    "adv_002": {"prior_violations": 0, "summary": "Established clean record."}
}

@app.get("/advertiser/{advertiser_id}")
def get_advertiser(advertiser_id: str):
    return ADVERTISERS.get(advertiser_id, {"prior_violations": 0, "summary": "New advertiser."})

@app.get("/health")
def health():
    return {"status": "ok", "service": "advertiser-crm"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)