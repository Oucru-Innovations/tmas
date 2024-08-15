@echo off
REM Change to the directory where the requirements.txt file is located
cd /d "%~dp0"

REM Install the required packages from requirements.txt
pip install -r requirements.txt

REM Install the current package in editable mode
pip install -e .
