import cv2
import numpy as np
from ultralytics import YOLO
import time

model = YOLO('yolo12n')

def detect_and_count_fruits(frame):
    fruit_classes = ['apple', 'banana', 'orange']
    
    fruit_counts = {fruit: 0 for fruit in fruit_classes}
    
    color_map = {
        'apple': (0, 0, 255), 
        'banana': (0, 255, 255), 
        'orange': (0, 165, 255) 
    }
    
    height, width = frame.shape[:2]
    
    max_dimension = 1280 
    scale = 1.0
    
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        inference_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    else:
        inference_frame = frame
    
    results = model(inference_frame)
    detections = results[0]  
    
    for detection in detections.boxes.data.tolist():
        xmin, ymin, xmax, ymax, confidence, class_id = detection[:6]
        class_name = model.names[int(class_id)]
        
        if class_name in fruit_classes:
            fruit_counts[class_name] += 1
            
            if scale != 1.0:
                xmin, ymin, xmax, ymax = [coord / scale for coord in [xmin, ymin, xmax, ymax]]
            
            color = color_map.get(class_name, (0, 255, 0))
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            
            label = f"{class_name} {confidence:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            cv2.rectangle(frame, (int(xmin), int(ymin - text_size[1] - 10)), 
                         (int(xmin + text_size[0]), int(ymin)), color, -1)
            
            cv2.putText(frame, label, (int(xmin), int(ymin - 5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    total_count = sum(fruit_counts.values())
    
    return frame, fruit_counts, total_count

def process_video(input_path, output_path, target_fruits=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video with resolution: {frame_width}x{frame_height} at {fps} FPS")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, fruit_counts, total_count = detect_and_count_fruits(frame)
        
        y_position = 30
        cv2.putText(processed_frame, f'Total: {total_count}', 
                   (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        for fruit, count in fruit_counts.items():
            if target_fruits is None or fruit in target_fruits:
                y_position += 40
                cv2.putText(processed_frame, f'{fruit}: {count}', 
                           (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(processed_frame)
        
        cv2.imshow('Frame', cv2.resize(processed_frame, (0, 0), fx=0.7, fy=0.7))
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps_actual = frame_count / elapsed_time
            print(f"Processed {frame_count} frames. FPS: {fps_actual:.2f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing complete. Output saved to {output_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")

if __name__ == "__main__":
    input_video = 'apple.mp4'
    output_video = 'output_highres.mp4'
    
    target_fruits = ['apple'] #target buah
    
    process_video(input_video, output_video, target_fruits)