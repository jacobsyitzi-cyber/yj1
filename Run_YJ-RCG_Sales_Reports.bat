@echo off
title YJ - RCG Sales Reports
cd /d "%~dp0"

if not exist .venv (
  py -m venv .venv
)

call .venv\Scripts\activate

REM Install only once (delete .installed to force reinstall)
if not exist .installed (
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  echo ok > .installed
)

REM Start the app (DO NOT build an EXE)
python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false

pause
