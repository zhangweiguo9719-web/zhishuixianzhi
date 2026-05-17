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
echo 智水先知 - 数据预处理
echo ============================================
echo [OK] 使用Python: %PYTHON_EXE%
echo [PREP] 执行数据预处理...

%PYTHON_EXE% preprocessing.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 数据预处理失败
    pause
    exit /b 1
)
echo.
echo [OK] 数据预处理完成！
pause
