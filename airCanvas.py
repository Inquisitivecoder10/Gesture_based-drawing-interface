import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Set up Streamlit
st.title("Air Canvas - Gesture-Based Drawing App")
st.write("Draw in the air and see your drawings appear on the canvas!")

# Configuration sliders for drawing
st.sidebar.title("Settings")
stroke_color = st.sidebar.color_picker("Stroke Color", "#00FF00")
stroke_size = st.sidebar.slider("Stroke Size", 1, 10, 5)

# Button to start/stop drawing
drawing_active = st.sidebar.checkbox("Enable Drawing", True)

# Initialize canvas
canvas = np.zeros((480, 640, 3), dtype="uint8")

# Main loop
cap = cv2.VideoCapture(0)  # Capture from webcam
if drawing_active:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture video.")
            break
        
        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get fingertip coordinates
                x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                # Draw circle at fingertip on canvas
                cv2.circle(canvas, (x, y), stroke_size, tuple(int(stroke_color[i:i+2], 16) for i in (1, 3, 5)), -1)
                
                # Draw hand landmarks on frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert frame and canvas to display in Streamlit
        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        st.image(combined, channels="RGB")
else:
    st.write("Drawing is currently disabled.")

cap.release()
hands.close()
