from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Compliance Audit API")
logs = []

class AuditRecord(BaseModel):
    ad_id: str
    action_taken: str
    reasoning: str

@app.post("/log")
def log_audit(record: AuditRecord):
    logs.append(record.dict())
    return {"status": "success", "audit_id": f"AUD-{len(logs)}"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "compliance-audit"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)