import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Copying Mechanism", layout="centered")
st.title("🧬 Copying Mechanism")

start = st.button("▶ Start")
stop = st.button("⏹ Stop")

frame_window = st.image([])

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands()

POSE_COLOR = (0, 255, 255)
HAND_COLOR = (0, 255, 255)
JOINT_COLOR = (255, 255, 255)

def draw_skeleton(img, landmarks, connections, color):
    h, w, _ = img.shape
    for start, end in connections:
        p1 = landmarks[start]
        p2 = landmarks[end]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (x, y), 4, JOINT_COLOR, -1)

if start:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb)
        hand_results = hands.process(rgb)

        if pose_results.pose_landmarks:
            draw_skeleton(frame, pose_results.pose_landmarks.landmark, mp_pose.POSE_CONNECTIONS, POSE_COLOR)

        if hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_hand_landmarks:
                draw_skeleton(frame, hand.landmark, mp_hands.HAND_CONNECTIONS, HAND_COLOR)

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if stop:
            break

    cap.release()
