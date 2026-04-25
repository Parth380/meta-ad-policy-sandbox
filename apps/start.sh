#!/bin/bash
set -e

python apps/regulatory_api.py &
REG_PID=$!
python apps/crm_api.py &
CRM_PID=$!
python apps/audit_api.py &
AUD_PID=$!

wait_for_service() {
  local url=$1
  local name=$2
  for i in $(seq 1 30); do
    if curl -sf "$url" > /dev/null 2>&1; then
      echo "[start.sh] $name ready"
      return 0
    fi
    sleep 1
  done
  echo "[start.sh] WARNING: $name did not become ready within 30s"
  return 1
}

wait_for_service "http://localhost:8001/health" "regulatory_api"
wait_for_service "http://localhost:8002/health" "crm_api"
wait_for_service "http://localhost:8003/health" "audit_api"

echo "[start.sh] All microservices up. Launching environment server on :8000"
exec uvicorn server.app:app --host 0.0.0.0 --port 8000
