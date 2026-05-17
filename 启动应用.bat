@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PYTHON_EXE=
where python >nul 2>&1
if %errorlevel% equ 0 set PYTHON_EXE=python
if "%PYTHON_EXE%"=="" (
    if exist "D:\Anaconda\python.exe" set PYTHON_EXE=D:\Anaconda\python.exe
    if exist "D:\anaconda3\python.exe" set PYTHON_EXE=D:\anaconda3\python.exe
)
if "%PYTHON_EXE%"=="" (
    echo [ERROR]找不到Python环境
    pause
    exit /b 1
)

echo ============================================
echo 智水先知AI-OPS泵站智控平台 - 启动界面
echo ============================================
echo [OK] 使用Python: %PYTHON_EXE%

if not exist "models\pump_lstm_final.pth" (
    echo [ERROR] 未找到模型文件，请先运行预处理和训练
    echo 运行: 运行预处理.bat
    echo 运行: 运行训练.bat
    pause
    exit /b 1
)
if not exist "models\scaler.pkl" (
    echo [ERROR] 未找到归一化器，请先运行预处理
    echo 运行: 运行预处理.bat
    pause
    exit /b 1
)

echo [OK] 模型文件检查通过
echo.

REM 自动选择空闲端口
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :850[0-9] ^| findstr LISTENING') do (
    set PORT_USED=%%a
)
if defined PORT_USED (
    echo [WARN] 端口8501已被占用，自动切换到8502
    set PORT=8502
) else (
    set PORT=8501
)

echo [START] 启动可视化界面...
echo 访问地址: http://localhost:%PORT%
echo 按Ctrl+C停止
echo.

%PYTHON_EXE% -m streamlit run app.py --server.headless true --server.port %PORT% --browser.gatherUsageStats false
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 启动失败
    pause
    exit /b 1
)
pause
