@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在检查 Python...
py --version >nul 2>&1
if errorlevel 1 (
  echo 未检测到 Python。请先安装 Python 3.11 或 3.12，并勾选 Add Python to PATH。
  pause
  exit /b 1
)
echo 正在安装或更新程序依赖，首次运行可能需要几分钟...
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
if errorlevel 1 (
  echo 依赖安装失败，请检查网络后重试。
  pause
  exit /b 1
)
echo 正在启动 A股智能K线AI...
py -m streamlit run app.py
pause
