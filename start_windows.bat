@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv
  if errorlevel 1 goto fail
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe app.py
pause
exit /b
:fail
echo Setup failed. Use Python 3.11 or 3.12 with Tkinter and check your internet connection.
pause
