from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import random
import uuid 

app = FastAPI(title="Compliance Audit API")
logs = []

class AuditRecord(BaseModel):
    ad_id: str
    action_taken: str
    reasoning: str

@app.post("/log")
def log_audit(record: AuditRecord):
    if random.random() < 0.1:
        return {"error": "service_unavailable", "retryable": True}
    audit_id = f"AUD-{uuid.uuid4().hex[:8].upper()}"
    logs.append({**record.dict(), "audit_id": audit_id})
    return {"status": "logged", "audit_id": audit_id}
@app.get("/health")
def health():
    return {"status": "ok", "service": "compliance-audit"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)