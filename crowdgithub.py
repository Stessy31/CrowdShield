from flask import Flask, render_template, request, redirect
import cv2
from ultralytics import YOLO
import psycopg2
import os
import threading
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client
import time

app = Flask(__name__)

# -----------------------------
# CONFIGURATION
# -----------------------------
app.secret_key = os.getenv("FLASK_SECRET_KEY", "development-secret-key")

# -----------------------------
# GLOBAL VARIABLES
# -----------------------------
user_email = None
user_phone = None
last_alert_time = 0

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

# -----------------------------
# POSTGRESQL CONNECTION
# -----------------------------
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME", "postgres"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432")
)
cursor = conn.cursor()

# -----------------------------
# ALERT CONFIGURATION
# -----------------------------
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Camera URL
CAMERA_URL = os.getenv(
    "CAMERA_URL",
    "http://127.0.0.1:8080/video"
)

# -----------------------------
# EMAIL FUNCTION
# -----------------------------
def send_email(to_email):
    try:
        print("📧 Sending email to:", to_email)

        msg = MIMEText("🚨 HIGH RISK CROWD DETECTED!")
        msg["Subject"] = "CrowdShield Alert"
        msg["From"] = EMAIL_SENDER
        msg["To"] = to_email

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(
            EMAIL_SENDER,
            EMAIL_PASSWORD.replace(" ", "").strip()
        )
        server.send_message(msg)
        server.quit()

        print("✅ Email sent!")

    except Exception as e:
        print("❌ Email error:", e)


# -----------------------------
# SMS FUNCTION
# -----------------------------
def send_sms(to_phone):
    try:
        print("📱 Sending SMS to:", to_phone)

        client = Client(TWILIO_SID, TWILIO_AUTH)

        message = client.messages.create(
            body="🚨 HIGH RISK CROWD DETECTED!",
            from_=TWILIO_NUMBER,
            to=to_phone
        )

        print("✅ SMS sent! SID:", message.sid)

    except Exception as e:
        print("❌ SMS error:", e)


# -----------------------------
# ALERT TRIGGER
# -----------------------------
def trigger_alert(risk):
    global last_alert_time, user_email, user_phone

    if risk == "HIGH RISK" and time.time() - last_alert_time > 60:
        print("🚨 ALERT TRIGGERED")
        print("Sending to:", user_email, user_phone)

        if not user_email or not user_phone:
            print("❌ ERROR: Email or Phone is None")
            return

        try:
            threading.Thread(
                target=send_email,
                args=(user_email,)
            ).start()

            threading.Thread(
                target=send_sms,
                args=(user_phone,)
            ).start()

            print("✅ Alert threads started")

        except Exception as e:
            print("❌ Thread error:", e)

        last_alert_time = time.time()


# -----------------------------
# AUTO-CALIBRATING DENSITY & RISK
# -----------------------------
def calculate_dynamic_density_and_risk(frame, results):
    total_frame_area = frame.shape[0] * frame.shape[1]
    people_boxes = []

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_area = (x2 - x1) * (y2 - y1)
                people_boxes.append(box_area)

    people_count = len(people_boxes)

    if people_count == 0:
        return 0, 0.0, "SAFE"

    people_boxes.sort()

    median_box_area = people_boxes[people_count // 2]

    estimated_capacity = (
        total_frame_area / max(1, median_box_area)
    ) * 0.40

    estimated_capacity = max(10, estimated_capacity)

    density_percentage = (
        people_count / estimated_capacity
    ) * 100

    density_percentage = min(100.0, density_percentage)

    if density_percentage < 50:
        risk = "SAFE"
    elif density_percentage < 85:
        risk = "WARNING"
    else:
        risk = "HIGH RISK"

    return people_count, round(density_percentage, 1), risk


# -----------------------------
# HOME
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# START
# -----------------------------
@app.route("/start", methods=["POST"])
def start():
    global user_email, user_phone

    user_email = request.form.get("email")
    user_phone = request.form.get("phone")

    print("✅ Saved email:", user_email)
    print("✅ Saved phone:", user_phone)

    return redirect("/live")


# -----------------------------
# VIDEO UPLOAD
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    global user_email, user_phone

    email = request.form.get("email")
    phone = request.form.get("phone")

    if email and phone:
        user_email = email
        user_phone = phone

    file = request.files["video"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    frame_id = 0

    print("📹 Processing video...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1
        frame_name = f"frame_{frame_id}"

        results = model(
            frame,
            conf=0.15,
            imgsz=1024
        )

        people_count, density, risk = (
            calculate_dynamic_density_and_risk(
                frame,
                results
            )
        )

        trigger_alert(risk)

        cursor.execute(
            """
            INSERT INTO crowd_data(
                frame_name,
                people_count,
                density,
                risk_level
            )
            VALUES (%s, %s, %s, %s)
            """,
            (
                frame_name,
                people_count,
                density,
                risk
            )
        )

        conn.commit()

    cap.release()

    print("✅ Video done")

    cursor.execute(
        "SELECT * FROM crowd_data "
        "ORDER BY id DESC LIMIT 20"
    )

    data = cursor.fetchall()

    return render_template(
        "index.html",
        data=data
    )


# -----------------------------
# FAST IP CAMERA STREAM
# -----------------------------
class VideoStream:

    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.grabbed, self.frame = (
            self.stream.read()
        )

        self.stopped = False

        self.thread = threading.Thread(
            target=self.update,
            args=()
        )

        self.thread.daemon = True
        self.thread.start()

    def update(self):

        while not self.stopped:

            if not self.grabbed:
                self.stopped = True

            else:
                self.grabbed, self.frame = (
                    self.stream.read()
                )

    def read(self):

        if self.frame is not None:
            return (
                self.grabbed,
                self.frame.copy()
            )

        return self.grabbed, self.frame

    def release(self):

        self.stopped = True

        if self.thread.is_alive():
            self.thread.join()

        self.stream.release()


# -----------------------------
# LIVE CAMERA FUNCTION
# -----------------------------
def start_live_camera():

    print("🎥 Camera started")

    cap = VideoStream(CAMERA_URL)

    time.sleep(1.0)

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            time.sleep(0.01)
            continue

        results = model(
            frame,
            conf=0.15,
            imgsz=640
        )

        people_count, density, risk = (
            calculate_dynamic_density_and_risk(
                frame,
                results
            )
        )

        trigger_alert(risk)

        cv2.putText(
            frame,
            f"Count: {people_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Density: {density}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Risk: {risk}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "Live Crowd Monitoring",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# LIVE ROUTE
# -----------------------------
@app.route("/live")
def live():

    global user_email, user_phone

    if not user_email or not user_phone:
        print(
            "❌ User data missing, "
            "redirecting to home"
        )

        return redirect("/")

    threading.Thread(
        target=start_live_camera
    ).start()

    return render_template("live.html")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)