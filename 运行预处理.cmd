@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PYTHON_EXE=
where python >nul 2>&1
if %errorlevel% equ 0 set PYTHON_EXE=python
if "%PYTHON_EXE%"=="" (
    if exist "D:\Anaconda\python.exe" set PYTHON_EXE=D:\Anaconda\python.exe
)
if "%PYTHON_EXE%"=="" (
    echo [ERROR] 找不到 Python 环境
    pause
    exit /b 1
)

%PYTHON_EXE% preprocessing.py
pause
