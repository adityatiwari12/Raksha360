import os
import cv2
import numpy as np
import streamlit as st
import av
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from PIL import Image
import time
from streamlit_notifications import notification
import pyttsx3

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

def speak_alert(message):
    tts_engine.say(message)
    tts_engine.runAndWait()

# Custom CSS for better UI
st.set_page_config(
    page_title="Raksha360 - Disaster Prediction & Response",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #ff4b4b, #ff9a5a);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .fire-detected {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        animation: alert 1s infinite;
    }
    @keyframes alert {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .no-fire {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="header">
    <h1>🔥 Raksha360</h1>
    <p>One-stop solution for Disaster Prediction & Response Ecosystem</p>
</div>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        from tensorflow.keras.models import load_model
        return load_model('fire_detection_model.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Define image size
img_size = (224, 224)

def preprocess_image(img):
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Video processing function
def process_video(video_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_bytes)
        video_path = tmpfile.name
    
    cap = cv2.VideoCapture(video_path)
    return cap

# WebRTC Video Processor
class VideoProcessor:
    def __init__(self):
        self.fire_detected = False
        self.last_alert_time = 0
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Make prediction
        processed_img = preprocess_image(img)
        pred = model.predict(processed_img)[0][0]
        
        # Draw prediction on frame
        if pred > 0.1:
            cv2.putText(img, "FIRE DETECTED!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (10, 10), (img.shape[1]-10, img.shape[0]-10), (0, 0, 255), 5)
            
            # Trigger alert every 10 seconds if fire is detected
            current_time = time.time()
            if current_time - self.last_alert_time > 10:
                self.fire_detected = True
                self.last_alert_time = current_time
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Home", "Image Detection", "Real-time Webcam", "Video Upload"])

if app_mode == "Home":
    st.markdown("""
    ## Welcome to Raksha360
    
    A comprehensive disaster prediction and response system that helps in early detection and alerting for fire hazards.
    
    ### Features:
    - 🔥 Fire detection in images
    - 📹 Real-time webcam monitoring
    - 🎥 Video file analysis
    - 🚨 Automatic alerts and notifications
    
    ### How to use:
    1. Select a mode from the sidebar
    2. Upload an image/video or use your webcam
    3. Get instant fire detection results
    """)

elif app_mode == "Image Detection":
    st.header("Image Fire Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        processed_img = preprocess_image(img)
        pred = model.predict(processed_img)[0][0]
        pred_label = 'Fire Detected!' if pred > 0.1 else 'No Fire Detected'
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption='Uploaded Image', use_column_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="prediction-box {'fire-detected' if pred > 0.1 else 'no-fire'}">
                <h3>{pred_label}</h3>
                <p>Confidence: {pred*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            if pred > 0.1:
                st.error("🚨 FIRE ALERT: Evacuate the area and call emergency services!")
                speak_alert("Warning! Fire detected. Please evacuate the area immediately.")
                notification("🚨 Fire Alert!", "Fire has been detected in the uploaded image.", duration=10)

elif app_mode == "Real-time Webcam":
    st.header("Real-time Webcam Detection")
    st.info("This feature uses your webcam for real-time fire detection.")
    
    webrtc_ctx = webrtc_streamer(
        key="fire-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if webrtc_ctx.video_processor:
        if webrtc_ctx.video_processor.fire_detected:
            st.error("🚨 FIRE DETECTED! Please take immediate action!")
            speak_alert("Warning! Fire detected in the webcam feed!")
            notification("🚨 Fire Alert!", "Fire detected in webcam feed!", duration=10)
            webrtc_ctx.video_processor.fire_detected = False

elif app_mode == "Video Upload":
    st.header("Video Fire Detection")
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    
    if video_file is not None:
        video_bytes = video_file.read()
        cap = process_video(video_bytes)
        
        stframe = st.empty()
        stop_button = st.button("Stop")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_img = preprocess_image(frame)
            pred = model.predict(processed_img)[0][0]
            
            # Draw prediction on frame
            if pred > 0.1:
                cv2.putText(frame, "FIRE DETECTED!", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, frame.shape[0]-10), (0, 0, 255), 5)
                
                # Show alert
                st.error("🚨 FIRE DETECTED IN VIDEO!")
                speak_alert("Warning! Fire detected in the video!")
                notification("🚨 Fire Alert!", "Fire detected in the video!", duration=10)
            
            # Display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", use_column_width=True)
            
            # Break the loop if stop button is pressed
            if stop_button:
                break
        
        cap.release()
        st.success("Video processing completed!")
