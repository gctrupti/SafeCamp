import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
ROI_POLYGON = [(100, 100), (500, 100), (500, 400), (100, 400)]  # Polygon for people counting
ALERTS_FILE = "alerts.json"
os.makedirs("alerts", exist_ok=True)

# YOLO models
person_model = YOLO("yolov8s.pt")        # COCO pretrained (person + bags)
fire_model = YOLO("bestfire.pt")         # YOLOv8 fire model
pose_model = YOLO("yolov8s-pose.pt")     # YOLOv8-pose model for fall detection
sharp_model = YOLO("best.pt")            # Custom model for sharp object detection

# Webcam
cap = cv2.VideoCapture(0)

# Cooldown to prevent log spamming
last_alert_time = None
ALERT_COOLDOWN = timedelta(seconds=5)

# Unattended bag tracking
UNATTENDED_THRESHOLD = timedelta(seconds=3)  # 3 seconds for unattended
unattended_bags = {}
bag_id_counter = 0

print("[INFO] Running Smart Surveillance System... Press 'q' to stop.")

# ---------------- FALL DETECTION FUNCTION ----------------
def is_fallen(keypoints, ratio_threshold=0.5):
    """
    Detect fall based on torso orientation.
    keypoints: Nx3 array (x, y, confidence) from YOLOv8-pose
    ratio_threshold: dy/dx ratio below which torso is considered horizontal
    """
    if keypoints is None or len(keypoints) < 17:
        return False

    # COCO keypoints indices: 5-left shoulder, 6-right shoulder, 11-left hip, 12-right hip
    left_shoulder = keypoints[5][:2]
    right_shoulder = keypoints[6][:2]
    left_hip = keypoints[11][:2]
    right_hip = keypoints[12][:2]

    torso_top = np.array([(left_shoulder[0] + right_shoulder[0]) / 2,
                          (left_shoulder[1] + right_shoulder[1]) / 2])
    torso_bottom = np.array([(left_hip[0] + right_hip[0]) / 2,
                             (left_hip[1] + right_hip[1]) / 2])

    dx = abs(torso_bottom[0] - torso_top[0])
    dy = abs(torso_bottom[1] - torso_top[1])

    # Horizontal torso detection
    return dy / (dx + 1e-6) < ratio_threshold

# ---------------- MAIN LOOP ----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = datetime.now()
    people_in_roi = 0
    fire_detected_frame = False
    fall_detected = False
    sharp_detected_frame = False
    people_centers = []
    bag_centers = []

    # ---------------- PERSON & BAG DETECTION ----------------
    results = person_model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = person_model.names[cls_id]
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            if label == "person":
                people_centers.append((cx, cy))
                # Count only if person is inside ROI
                if ROI_POLYGON[0][0] < cx < ROI_POLYGON[1][0] and ROI_POLYGON[0][1] < cy < ROI_POLYGON[2][1]:
                    people_in_roi += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            elif label in ["backpack", "handbag", "suitcase"]:
                bag_centers.append((cx, cy, x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # ---------------- FALL DETECTION USING POSE ----------------
    pose_results = pose_model(frame, verbose=False)
    for r in pose_results:
        if r.boxes is not None and r.keypoints is not None:
            for kpts in r.keypoints:
                if is_fallen(kpts):
                    fall_detected = True
                    x1, y1, x2, y2 = map(int, r.boxes.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                    cv2.putText(frame, "Fall!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # ---------------- UNATTENDED BAG LOGIC ----------------
    for cx, cy, x1, y1, x2, y2 in bag_centers:
        nearby = any(np.hypot(cx - px, cy - py) < 100 for px, py in people_centers)
        found = False

        for bag_id, data in unattended_bags.items():
            bx, by, bx2, by2 = data["bbox"]
            b_cx = (bx + bx2) // 2
            b_cy = (by + by2) // 2

            if abs(b_cx - cx) < 50 and abs(b_cy - cy) < 50:
                found = True
                unattended_bags[bag_id]["bbox"] = (x1, y1, x2, y2)

                if nearby:
                    unattended_bags[bag_id]["unattended"] = False
                    if data["unattended"]:
                        unattended_bags[bag_id]["first_seen"] = now
                else:
                    if not data["unattended"] and now - data["first_seen"] >= UNATTENDED_THRESHOLD:
                        unattended_bags[bag_id]["unattended"] = True
                        cv2.putText(frame, "Unattended", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                break

        if not found:
            bag_id_counter += 1
            unattended_bags[bag_id_counter] = {
                "bbox": (x1, y1, x2, y2),
                "first_seen": now,
                "unattended": False
            }

    # Remove disappeared bags
    current_bags = [(bx, by, bx2, by2) for _, (cx, cy, bx, by, bx2, by2) in enumerate(bag_centers)]
    to_remove = []
    for bag_id, data in unattended_bags.items():
        bx, by, bx2, by2 = data["bbox"]
        if not any(abs((bx+bx2)//2 - (cx+bx2)//2) < 50 and abs((by+by2)//2 - (cy+by2)//2) < 50 for cx, cy, bx2, by2 in current_bags):
            to_remove.append(bag_id)
    for bag_id in to_remove:
        unattended_bags.pop(bag_id)

    # ---------------- FIRE DETECTION ----------------
    fire_results = fire_model(frame, verbose=False)
    for r in fire_results:
        for box in r.boxes:
            if fire_model.names[int(box.cls)].lower() == "fire":
                fire_detected_frame = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Fire", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # ---------------- SHARP OBJECT DETECTION ----------------
    sharp_results = sharp_model(frame, verbose=False)
    for r in sharp_results:
        for box in r.boxes:
            label = sharp_model.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            sharp_detected_frame = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,255), 2)
            cv2.putText(frame, f"Sharp: {label}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

    # ---------------- ALERT DECISION ----------------
    alert_needed = (
        people_in_roi > 0 or
        any(bag["unattended"] for bag in unattended_bags.values()) or
        fire_detected_frame or
        fall_detected or
        sharp_detected_frame
    )

    # ---------------- ALERT LOGGING ----------------
    if alert_needed and (last_alert_time is None or now - last_alert_time > ALERT_COOLDOWN):
        ts = now.strftime("%Y%m%d_%H%M%S")
        img_path = f"alerts/Alert_{ts}.jpg"
        cv2.imwrite(img_path, frame)

        alert = {
            "timestamp": now.isoformat(),
            "num_people": people_in_roi,
            "unattended_bags": sum(1 for b in unattended_bags.values() if b["unattended"]),
            "fire_detected": fire_detected_frame,
            "fall_detected": fall_detected,
            "sharp_detected": sharp_detected_frame,
            "image": img_path
        }

        if os.path.exists(ALERTS_FILE) and os.path.getsize(ALERTS_FILE) > 0:
            with open(ALERTS_FILE, "r") as f:
                try:
                    alerts = json.load(f)
                except json.JSONDecodeError:
                    alerts = []
        else:
            alerts = []

        alerts.append(alert)
        with open(ALERTS_FILE, "w") as f:
            json.dump(alerts, f, indent=4)

        print(f"[ALERT] People: {people_in_roi}, Unattended Bags: {alert['unattended_bags']}, Fire: {fire_detected_frame}, Fall: {fall_detected}, Sharp: {sharp_detected_frame} at {alert['timestamp']}")
        last_alert_time = now

    # Draw ROI
    cv2.polylines(frame, [np.array(ROI_POLYGON, np.int32)], isClosed=True, color=(0,255,0), thickness=2)

    cv2.imshow("Smart Surveillance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
