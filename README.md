# 🛡️ CrowdShield – AI-Based Crowd Monitoring System

CrowdShield is an AI-powered crowd monitoring system designed to analyze video streams, detect people, estimate crowd density, and identify potentially risky crowd situations using computer vision.

## 🚀 Features

- 👥 Person detection using YOLOv8
- 📊 Dynamic crowd density estimation
- 🟢 SAFE, 🟡 WARNING, and 🔴 HIGH RISK classification
- 🎥 Live camera monitoring
- 📤 Video upload and analysis
- 🔥 Crowd density heatmap generation
- 📧 Email alerts for high-risk situations
- 📱 SMS alerts using Twilio
- 🗄️ PostgreSQL-based detection logging
- 🌐 Flask-based web dashboard

## 🛠️ Technologies Used

- Python
- Flask
- YOLOv8
- OpenCV
- PostgreSQL
- Twilio
- HTML & CSS
- NumPy

## ⚙️ How It Works

CrowdShield takes a video or live camera stream as input and uses YOLOv8 to detect people in each frame. The detected people and their bounding boxes are used to estimate crowd density. Based on the estimated density, the system classifies the crowd as **SAFE**, **WARNING**, or **HIGH RISK**.

When a high-risk condition is detected, the system can trigger email and SMS alerts. Detection information such as the number of people, density, and risk level is stored in PostgreSQL for monitoring and analysis. The system also generates heatmaps to visualize areas with higher crowd concentration.

The project is designed as a prototype for intelligent crowd monitoring in **public gatherings, events, transportation areas, commercial spaces, and other high-traffic locations**.

## 📂 Project Structure

```text
CrowdShield/
│
├── crowdgithub.py
├── index.html
├── live.html
├── tiled_detector.py
└── README.md
