@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "..\rgbd_odometry_project\.venv\Scripts\python.exe" (
    start "" "http://127.0.0.1:8765/demo/index.html"
    "..\rgbd_odometry_project\.venv\Scripts\python.exe" src\serve_demo.py --root . --port 8765
) else (
    start "" "http://127.0.0.1:8765/demo/index.html"
    python src\serve_demo.py --root . --port 8765
)
