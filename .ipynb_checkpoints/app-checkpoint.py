import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLOv8 model from the hub
model = YOLO('yolov8n')

# Function to detect and count fruits in a video frame
def detect_and_count_fruits(frame):
    # List of fruit classes to detect (apple, banana, orange)
    fruit_classes = ['apple', 'banana', 'orange']
    
    # Dictionary to store counts for each fruit type
    fruit_counts = {fruit: 0 for fruit in fruit_classes}
    
    # Color map for different fruits (BGR format)
    color_map = {
        'apple': (0, 0, 255),    # Red
        'banana': (0, 255, 255), # Yellow
        'orange': (0, 165, 255)  # Orange
    }
    
    # Resize image for faster processing if needed
    # For high resolution videos, we can resize for inference and draw on original
    height, width = frame.shape[:2]
    
    # Determine if we need to resize based on image dimensions
    max_dimension = 1280  # Maximum dimension for reasonable processing speed
    scale = 1.0
    
    # If image is very large, resize it for inference
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        inference_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    else:
        inference_frame = frame
    
    # Perform inference on the frame
    results = model(inference_frame)
    detections = results[0]  # Get detections from the first item in results
    
    # Iterate over all detections
    for detection in detections.boxes.data.tolist():
        xmin, ymin, xmax, ymax, confidence, class_id = detection[:6]
        class_name = model.names[int(class_id)]
        
        # If detected object is one of our target fruits
        if class_name in fruit_classes:
            # Increment counter for this fruit type
            fruit_counts[class_name] += 1
            
            # If we resized the image, adjust the bounding box coordinates
            if scale != 1.0:
                xmin, ymin, xmax, ymax = [coord / scale for coord in [xmin, ymin, xmax, ymax]]
            
            # Draw bounding box around detected fruit with appropriate color
            color = color_map.get(class_name, (0, 255, 0))  # Default to green if not in map
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            
            # Draw class name and confidence
            label = f"{class_name} {confidence:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Draw background for text
            cv2.rectangle(frame, (int(xmin), int(ymin - text_size[1] - 10)), 
                         (int(xmin + text_size[0]), int(ymin)), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (int(xmin), int(ymin - 5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Total count of all fruits
    total_count = sum(fruit_counts.values())
    
    return frame, fruit_counts, total_count

# Function to process video
def process_video(input_path, output_path, target_fruits=None):
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video with resolution: {frame_width}x{frame_height} at {fps} FPS")
    
    # Define codec and create VideoWriter object to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and count fruits in the frame
        processed_frame, fruit_counts, total_count = detect_and_count_fruits(frame)
        
        # Display fruit counts on the frame
        y_position = 30
        cv2.putText(processed_frame, f'Total Fruits: {total_count}', 
                   (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display individual fruit counts
        for fruit, count in fruit_counts.items():
            if target_fruits is None or fruit in target_fruits:
                y_position += 40
                cv2.putText(processed_frame, f'{fruit}: {count}', 
                           (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write the processed frame to the output video
        out.write(processed_frame)
        
        # Display the frame with detections (optional, can be commented out for faster processing)
        cv2.imshow('Frame', cv2.resize(processed_frame, (0, 0), fx=0.7, fy=0.7))
        
        # Count processed frames
        frame_count += 1
        
        # Calculate and display FPS every 30 frames
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps_actual = frame_count / elapsed_time
            print(f"Processed {frame_count} frames. FPS: {fps_actual:.2f}")
        
        # Press 'q' to exit the video display early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing complete. Output saved to {output_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")

# Main execution
if __name__ == "__main__":
    # Path to input and output videos
    input_video = 'video5.mp4'  # Change this to your high-resolution video path
    output_video = 'output_highres.mp4'
    
    # Specify which fruits to detect and count (None for all supported fruits)
    # Options: 'apple', 'banana', 'orange'
    target_fruits = ['banana']  # Change this as needed or set to None for all fruits
    
    # Process the video
    process_video(input_video, output_video, target_fruits)