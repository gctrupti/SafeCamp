import subprocess
import time

# Start detection system
detection = subprocess.Popen(["python", "persion.py"])

# Wait a little before starting frontend
time.sleep(3)

# Start Streamlit frontend
frontend = subprocess.Popen(["python", "-m", "streamlit", "run", "output.py"])

try:
    detection.wait()
    frontend.wait()
except KeyboardInterrupt:
    detection.terminate()
    frontend.terminate()
