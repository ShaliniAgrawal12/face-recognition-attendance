import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

def mark_attendance(name, attendance_file='attendance.csv'):
    """Mark attendance in CSV file"""
    # Create attendance file with header if it doesn't exist
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,Date,Time')
    
    # Read existing names to avoid duplicates
    with open(attendance_file, 'r+') as f:
        data = f.readlines()
        names = [line.split(',')[0] for line in data]
        
        # Only add if name not already marked today
        if name not in names:
            now = datetime.now()
            date_str = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date_str},{time_str}')
            print(f"Marked attendance for {name} on {date_str} at {time_str}")
            return True
        else:
            print(f"Attendance already marked for {name}")
            return False

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created '{directory}' directory.")

def capture_and_save_face(name):
    """Capture face from webcam and save to images directory"""
    # Ensure images directory exists
    images_dir = 'images'
    ensure_directory(images_dir)
    
    # Initialize webcam
    print("Starting webcam to capture your face...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    face_detected = False
    saved_image_path = None
    
    try:
        while not face_detected:
            # Read frame from webcam
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame from camera")
                break
            
            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Display info and instructions
            display_frame = frame.copy()
            cv2.putText(display_frame, "Position your face in the center", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, "Press 'c' to capture when ready", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, "Press 'q' to quit", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw rectangle around detected faces
            for top, right, bottom, left in face_locations:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Capture Face', display_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            # If 'c' is pressed and a face is detected, save the image
            if key == ord('c') and face_locations:
                # Create filename with name and timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{name.lower()}_{timestamp}.jpg"
                image_path = os.path.join(images_dir, filename)
                
                # Save the image
                cv2.imwrite(image_path, frame)
                print(f"Face captured and saved as {image_path}")
                
                # Set flag to exit loop
                face_detected = True
                saved_image_path = image_path
            
            # If 'q' is pressed, exit
            elif key == ord('q'):
                print("Face capture cancelled")
                return False
    
    except Exception as e:
        print(f"Error during face capture: {e}")
        return False
        
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    return saved_image_path

def verify_face_encoding(image_path):
    """Verify that face can be detected and encoded in the saved image"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return False
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            print(f"Error: No face detected in saved image")
            return False
        
        # Get face encoding
        face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
        print("Face successfully encoded")
        return True
        
    except Exception as e:
        print(f"Error verifying face encoding: {e}")
        return False

def main():
    print("=== Simple Attendance System ===")
    
    # Ask for user's name
    name = input("Please enter your name: ").strip()
    if not name:
        print("Name cannot be empty. Exiting.")
        return
    
    # Capitalize name for consistency
    name = name.upper()
    
    # Capture and save face
    image_path = capture_and_save_face(name)
    if not image_path:
        print("Face capture failed. Exiting.")
        return
    
    # Verify face encoding
    if not verify_face_encoding(image_path):
        print("Face verification failed. Please try again with better lighting and positioning.")
        return
    
    # Mark attendance
    mark_attendance(name)
    
    print(f"Attendance process completed successfully for {name}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")