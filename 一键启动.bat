@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PYTHON_EXE=
where python >nul 2>&1
if %errorlevel% equ 0 set PYTHON_EXE=python
if "%PYTHON_EXE%"=="" (
    if exist "D:\Anaconda\python.exe" set PYTHON_EXE=D:\Anaconda\python.exe
    if exist "D:\anaconda3\python.exe" set PYTHON_EXE=D:\anaconda3\python.exe
    if exist "%USERPROFILE%\anaconda3\python.exe" set PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe
)
if "%PYTHON_EXE%"=="" (
    echo [ERROR]找不到Python环境，请安装Python或Anaconda
    pause
    exit /b 1
)

echo ============================================
echo 智水先知AI-OPS泵站智控平台 - 一键启动
echo ============================================
echo [OK] 使用Python: %PYTHON_EXE%

echo.
echo [1/4] 检查依赖包...
%PYTHON_EXE% -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [INSTALL] 正在安装streamlit...
    %PYTHON_EXE% -m pip install streamlit -q
)
echo [OK] streamlit已就绪

echo.
echo [2/4] 检查数据文件...
if not exist "data\real_pump_data.csv" (
    echo [GEN] 数据文件不存在，正在生成...
    %PYTHON_EXE% forge_big_data.py
) else (
    echo [OK] 数据文件已存在
)

echo.
echo [3/4] 检查预处理文件...
if not exist "models\scaler.pkl" (
    echo [PREP] 执行数据预处理...
    %PYTHON_EXE% preprocessing.py
    if %errorlevel% neq 0 (
        echo [ERROR] 数据预处理失败
        pause
        exit /b 1
    )
) else (
    echo [OK] 预处理文件已存在
)

echo.
echo [4/4] 检查模型文件...
if not exist "models\pump_lstm_final.pth" (
    echo [TRAIN] 正在训练LSTM模型，预计需要3-5分钟...
    %PYTHON_EXE% train.py
    if %errorlevel% neq 0 (
        echo [ERROR] 模型训练失败
        pause
        exit /b 1
    )
) else (
    echo [OK] 模型文件已存在
)

echo.
echo ============================================
echo 系统准备就绪，正在启动可视化界面...
echo 访问地址: http://localhost:8501
echo 按Ctrl+C停止系统
echo ============================================
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

%PYTHON_EXE% -m streamlit run app.py --server.headless true --server.port %PORT% --browser.gatherUsageStats false
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 启动失败，端口%PORT%可能也被占用
    echo 请关闭其他Streamlit应用后重试
    pause
    exit /b 1
)
pause
