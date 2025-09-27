import os
import json
import time
from pathlib import Path
import streamlit as st
from PIL import Image

# ---------------- CONFIG ----------------
ALERTS_FILE = Path("alerts.json")
ALERTS_DIR = Path("alerts")

st.set_page_config(page_title="Campus Security Alerts", layout="wide")

# ---------------- Helper Functions ----------------
def load_alerts():
    """Load alerts list from JSON file."""
    if not ALERTS_FILE.exists():
        return []
    try:
        with open(ALERTS_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data
    except Exception:
        return []

def normalize_image_path(alert):
    """Ensure correct path for snapshot image."""
    image_path = alert.get("image")
    if not image_path:
        return None
    p = Path(image_path)
    if p.exists():
        return p.as_posix()
    alt = ALERTS_DIR / p.name
    if alt.exists():
        return alt.as_posix()
    return None

# ---------------- Streamlit UI ----------------
st.title("🚨 Automated Campus Security Alerts")
st.info("This dashboard updates automatically when new alerts are added.")

# Session state to track last alert count
if "last_alert_count" not in st.session_state:
    st.session_state.last_alert_count = 0

# Auto-refresh checkbox
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh every (seconds)", 2, 20, 5)

alerts = load_alerts()

# If new alerts are added, update session state
if len(alerts) > st.session_state.last_alert_count:
    st.session_state.last_alert_count = len(alerts)

# Show all alerts as stacked cards (latest on top)
if alerts:
    for alert in reversed(alerts):  # latest first
        with st.container():
            st.subheader("⚠️ Alert")
            st.write(f"**Time:** {alert.get('timestamp', 'Unknown')}")
            st.write(f"**People in ROI:** {alert.get('num_people', 'N/A')}")
            st.write(f"**Unattended Bags:** {alert.get('unattended_bags', 'N/A')}")
            st.write(f"**Fire Detected:** {alert.get('fire_detected', False)}")
            st.write(f"**Fall Detected:** {alert.get('fall_detected', False)}")
            st.write(f"**Sharp Object Detected:** {alert.get('sharp_detected', False)}")

            img_path = normalize_image_path(alert)
            if img_path:
                st.image(Image.open(img_path), caption="Alert Snapshot", use_column_width=True)

            st.code(json.dumps(alert, indent=2), language="json")
            st.markdown("---")  # separator line

else:
    st.success("✅ No alerts yet.")

# Auto-refresh loop
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
