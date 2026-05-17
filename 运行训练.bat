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
    echo [ERROR]找不到Python环境
    pause
    exit /b 1
)

echo ============================================
echo 智水先知 - LSTM模型训练
echo ============================================
echo [OK] 使用Python: %PYTHON_EXE%
echo [TRAIN] 开始训练，预计3-5分钟...

%PYTHON_EXE% train.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 训练失败
    pause
    exit /b 1
)
echo.
echo [OK] 训练完成！
pause
