#!/bin/bash

# Start the background microservices
python apps/regulatory_api.py &
python apps/crm_api.py &
python apps/audit_api.py &

# Start the main environment server in the foreground
uvicorn server.app:app --host 0.0.0.0 --port 8000