import os
import sys
import subprocess
import webbrowser
import time
import socket

APP_PORT = 8501

def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    app_py = os.path.join(here, "app.py")

    cmd = [
        sys.executable, "-m", "streamlit", "run", app_py,
        "--server.headless", "true",
        "--server.port", str(APP_PORT),
        "--browser.gatherUsageStats", "false",
    ]
    proc = subprocess.Popen(cmd, cwd=here)

    for _ in range(300):
        if port_open(APP_PORT):
            webbrowser.open(f"http://localhost:{APP_PORT}")
            break
        time.sleep(0.1)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()

if __name__ == "__main__":
    main()
