# YJ - RCG Sales Reports (Clean rebuild - no packaging errors)

This is the correct structure for turning a Streamlit dashboard into a Windows "program":

- `app.py` = the Streamlit dashboard (must be launched via Streamlit)
- `launcher.py` = starts Streamlit and opens your browser automatically
- `Run_YJ-RCG_Sales_Reports.bat` = simplest double-click launcher (recommended)
- `Build_EXE.bat` = builds an EXE folder that launches the app correctly

## Option B (recommended)
Double-click: `Run_YJ-RCG_Sales_Reports.bat`

## Build a Windows program folder (EXE)
Double-click: `Build_EXE.bat`
Then run: `dist\YJ-RCG_Sales_Reports\YJ-RCG_Sales_Reports.exe`

## Defender note
PyInstaller EXEs are often flagged. Add this folder to Windows Defender exclusions before building.
