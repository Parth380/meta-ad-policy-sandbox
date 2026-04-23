@echo off
echo Launching Enterprise Ecosystem...

:: This line forces Windows to go to the project root before running anything
cd /d "%~dp0\.."

start "Regulatory API" cmd /k "uv run python apps\regulatory_api.py"
start "CRM API" cmd /k "uv run python apps\crm_api.py"
start "Audit API" cmd /k "uv run python apps\audit_api.py"
start "Environment Server" cmd /k "uv run uvicorn server.app:app --host 0.0.0.0 --port 8000"