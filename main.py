from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import cv2
import dlib
import math
import numpy as np
from deepface import DeepFace
import io
from typing import Dict
import os
import base64

app = FastAPI(title="Liveness Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables and models
detector = dlib.get_frontal_face_detector()

predictor_path = "model68/shape_predictor_68_face_landmarks.dat"

# Download the shape predictor if it doesn't exist
if not os.path.exists(predictor_path):
    raise RuntimeError("Please download the shape_predictor_68_face_landmarks.dat file and place it in the model68 directory")

predictor = dlib.shape_predictor(predictor_path)

def calculate_face_size(landmarks, frame):
    """Calculate the bounding box size of the face in pixels."""
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]
    
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    face_width = max_x - min_x
    face_height = max_y - min_y
    
    face_area = face_width * face_height
    return face_area

def calculate_ear(eye_points):
    """Calculate the eye aspect ratio."""
    A = math.dist(eye_points[1], eye_points[5])
    B = math.dist(eye_points[2], eye_points[4])
    C = math.dist(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blink(landmarks):
    """Detect if the person is blinking."""
    left_eye = [(landmarks[i].x, landmarks[i].y) for i in range(36, 42)]
    right_eye = [(landmarks[i].x, landmarks[i].y) for i in range(42, 48)]
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    return {
        "is_blinking": left_ear < 0.22 or right_ear < 0.22,
        "left_ear": left_ear,
        "right_ear": right_ear
    }

def process_image(image_bytes: bytes) -> Dict:
    """Process the uploaded image and perform liveness detection."""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    try:
        # Extract faces using DeepFace
        face_objs = DeepFace.extract_faces(img_path=frame, enforce_detection=False, anti_spoofing=True)
        
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        frame_height, frame_width, _ = frame.shape
        total_area = frame_width * frame_height

        for face_obj, face in zip(face_objs, faces):
            if 'facial_area' not in face_obj:
                continue

            facial_area = face_obj['facial_area']
            is_real = face_obj.get("is_real", False)
            
            # Get facial landmarks
            landmarks = predictor(gray, face).parts()
            
            # Calculate metrics
            face_size = calculate_face_size(landmarks, frame)
            face_percentage = (face_size / total_area) * 100
            blink_info = detect_blink(landmarks)
            
            result = {
                "is_real": is_real,
                "face_percentage": face_percentage,
                "blink_detected": blink_info["is_blinking"],
                "left_ear": blink_info["left_ear"],
                "right_ear": blink_info["right_ear"],
                "facial_area": facial_area,
                "confidence": face_obj.get("confidence", 0),
            }
            
            # Apply liveness detection rules
            result["final_liveness_status"] = (
                is_real and 
                (
                    (face_percentage > 25 and face_percentage < 29 and blink_info["is_blinking"]) or
                    (face_percentage > 30 and blink_info["is_blinking"]) or
                    (face_percentage < 26 and face_percentage > 20 and blink_info["is_blinking"]) or
                    (face_percentage < 20 and face_percentage > 16 and blink_info["is_blinking"]) or
                    (face_percentage < 18 and face_percentage > 11 and blink_info["is_blinking"]) or
                    (face_percentage > 7 and face_percentage < 11 and blink_info["is_blinking"]) or
                    (face_percentage < 6 and face_percentage > 2 and blink_info["is_blinking"])
                )
            )
            
            results.append(result)
        
        return {
            "success": True,
            "faces_detected": len(results),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect-liveness")
async def detect_liveness(file: UploadFile = File(...)):
    """
    Endpoint to detect liveness in an uploaded image.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    return process_image(contents)

@app.post("/detect-liveness-base64")
async def detect_liveness_base64(image_data: dict = Body(...)):
    """
    Endpoint to detect liveness in a base64 encoded image.
    """
    try:
        # Get base64 string from request body
        base64_image = image_data.get("image")
        if not base64_image:
            raise HTTPException(status_code=400, detail="No image data provided")

        # Remove data URL prefix if present
        if "base64," in base64_image:
            base64_image = base64_image.split("base64,")[1]

        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_image)
        
        return process_image(image_bytes)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing base64 image: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {"status": "Liveness Detection API is running"} 