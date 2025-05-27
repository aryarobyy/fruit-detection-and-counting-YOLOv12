import cv2
import numpy as np
from ultralytics import YOLO
import time
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

class HybridAppleDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.3):
        """
        Hybrid detector using YOLO for apple detection + color analysis for ripeness
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
        try:
            self.model = YOLO(model_path)
            model_classes = list(self.model.names.values())
            print(f"✅ Loaded model: {model_path}")
            print(f"📋 Available classes: {model_classes}")
            
            if 'apple' not in model_classes:
                print("⚠️  Warning: 'apple' class not found in model")
                print(f"Available classes: {model_classes}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Trying to download YOLOv8n...")
            self.model = YOLO('yolov8n.pt')
        
        self.color_map = {
            'ripe': (0, 255, 0), 
            'unripe': (0, 0, 255),
            'unknown': (255, 0, 0)
        }
    
    def classify_apple_ripeness(self, frame, bbox):
        """
        Enhanced color-based ripeness classification
        Returns: 'ripe', 'unripe', or 'unknown'
        """
        xmin, ymin, xmax, ymax = bbox
        
        h, w = frame.shape[:2]
        xmin = max(0, min(int(xmin), w-1))
        ymin = max(0, min(int(ymin), h-1))
        xmax = max(xmin+1, min(int(xmax), w))
        ymax = max(ymin+1, min(int(ymax), h))
        
        apple_region = frame[ymin:ymax, xmin:xmax]
        
        if apple_region.size == 0:
            return 'unknown'
        
        hsv = cv2.cvtColor(apple_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(apple_region, cv2.COLOR_BGR2LAB)
        
        red_lower1 = np.array([0, 40, 40])
        red_upper1 = np.array([15, 255, 255])
        red_lower2 = np.array([165, 40, 40])
        red_upper2 = np.array([180, 255, 255])
        
        yellow_lower = np.array([10, 40, 40])
        yellow_upper = np.array([40, 255, 255])
        
        green_lower = np.array([35, 30, 30])
        green_upper = np.array([85, 255, 255])
        
        dark_green_lower = np.array([40, 50, 20])
        dark_green_upper = np.array([80, 255, 120])
        
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        dark_green_mask = cv2.inRange(hsv, dark_green_lower, dark_green_upper)
        
        total_pixels = apple_region.shape[0] * apple_region.shape[1]
        red_percentage = np.sum(red_mask > 0) / total_pixels
        yellow_percentage = np.sum(yellow_mask > 0) / total_pixels
        green_percentage = np.sum(green_mask > 0) / total_pixels
        dark_green_percentage = np.sum(dark_green_mask > 0) / total_pixels
        
        mean_hue = np.mean(hsv[:, :, 0])
        mean_saturation = np.mean(hsv[:, :, 1])
        mean_brightness = np.mean(hsv[:, :, 2])
        
        mean_a = np.mean(lab[:, :, 1]) 
        mean_b = np.mean(lab[:, :, 2])
        
        ripe_score = 0
        unripe_score = 0
        
        if red_percentage > 0.2:
            ripe_score += 3
        elif red_percentage > 0.1:
            ripe_score += 1
        
        if yellow_percentage > 0.3:
            ripe_score += 3
        elif yellow_percentage > 0.15:
            ripe_score += 2
        
        if dark_green_percentage > 0.4:
            unripe_score += 3
        elif dark_green_percentage > 0.2:
            unripe_score += 2
        
        if mean_brightness > 100 and mean_saturation > 50:
            ripe_score += 1
        elif mean_brightness < 80:
            unripe_score += 1
        
        if mean_a > 135:
            ripe_score += 2
        elif mean_a < 120:
            unripe_score += 1
        
        if mean_b > 135:
            ripe_score += 1
        
        if (mean_hue < 20 or mean_hue > 160) and mean_saturation > 40:
            ripe_score += 2
        elif 20 < mean_hue < 40 and mean_saturation > 40:
            ripe_score += 1
        elif 40 < mean_hue < 80 and mean_brightness < 100:
            unripe_score += 2
        
        if ripe_score > unripe_score and ripe_score >= 2:
            return 'ripe'
        elif unripe_score > ripe_score and unripe_score >= 2:
            return 'unripe'
        elif green_percentage > 0.5 and mean_brightness > 120:
            return 'ripe'
        else:
            if red_percentage + yellow_percentage > green_percentage:
                return 'ripe' if red_percentage + yellow_percentage > 0.1 else 'unknown'
            else:
                return 'unripe' if green_percentage > 0.2 else 'unknown'
    
    def detect_and_classify_apples(self, frame):
        """
        Detect apples using YOLO and classify ripeness using color analysis
        """
        apple_counts = {'ripe': 0, 'unripe': 0, 'unknown': 0, 'total': 0}
        detections_info = []
        
        height, width = frame.shape[:2]
        max_dimension = 1280
        scale = 1.0
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            inference_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            inference_frame = frame
        
        try:
            results = self.model(inference_frame, conf=self.confidence_threshold, verbose=False)
            
            if len(results) > 0 and results[0].boxes is not None:
                detections = results[0].boxes.data.tolist()
                
                print(f"🔍 Found {len(detections)} detections in frame")
                
                for detection in detections:
                    if len(detection) >= 6:
                        xmin, ymin, xmax, ymax, confidence, class_id = detection[:6]
                        class_name = self.model.names[int(class_id)]
                        
                        print(f"   - Detected: {class_name} (confidence: {confidence:.3f})")
                        
                        if class_name == 'apple':
                            if scale != 1.0:
                                xmin, ymin, xmax, ymax = [coord / scale for coord in [xmin, ymin, xmax, ymax]]
                            
                            ripeness = self.classify_apple_ripeness(frame, (xmin, ymin, xmax, ymax))
                            
                            apple_counts[ripeness] += 1
                            apple_counts['total'] += 1
                            
                            print(f"     └─ Classified as: {ripeness}")
                            
                            detections_info.append({
                                'bbox': (xmin, ymin, xmax, ymax),
                                'confidence': confidence,
                                'class': 'apple',
                                'ripeness': ripeness
                            })
                            
                            color = self.color_map[ripeness]
                            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 3)
                            
                            label = f"{ripeness} apple {confidence:.2f}"
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            
                            cv2.rectangle(frame, (int(xmin), int(ymin - text_size[1] - 15)), 
                                        (int(xmin + text_size[0] + 10), int(ymin)), color, -1)
                            
                            cv2.putText(frame, label, (int(xmin + 5), int(ymin - 8)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                print("🔍 No detections found in frame")
                
        except Exception as e:
            print(f"❌ Error in detection: {e}")
        
        return frame, apple_counts, detections_info

class RipenessEvaluator:
    def __init__(self):
        self.all_predictions = []
        self.all_ground_truth = []
        self.frame_stats = []
        self.detection_details = []
    
    def add_frame_predictions(self, detections_info, ground_truth=None, frame_number=None):
        """Add predictions for a frame"""
        frame_predictions = [det['ripeness'] for det in detections_info if det['ripeness'] != 'unknown']
        frame_confidences = [det['confidence'] for det in detections_info if det['ripeness'] != 'unknown']
        
        ripe_count = sum(1 for det in detections_info if det['ripeness'] == 'ripe')
        unripe_count = sum(1 for det in detections_info if det['ripeness'] == 'unripe')
        unknown_count = sum(1 for det in detections_info if det['ripeness'] == 'unknown')
        
        self.frame_stats.append({
            'frame_number': frame_number,
            'ripe_count': ripe_count,
            'unripe_count': unripe_count,
            'unknown_count': unknown_count,
            'total_count': len(detections_info),
            'avg_confidence': np.mean(frame_confidences) if frame_confidences else 0
        })
        
        if ground_truth is not None:
            self.all_predictions.extend(frame_predictions)
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        if not self.all_predictions or len(set(self.all_predictions)) < 2:
            return None
        
        return {
            'total_samples': len(self.all_predictions),
            'ripe_count': self.all_predictions.count('ripe'),
            'unripe_count': self.all_predictions.count('unripe')
        }
    
    def get_summary_statistics(self):
        """Get comprehensive summary statistics"""
        if not self.frame_stats:
            return None
        
        total_ripe = sum(frame['ripe_count'] for frame in self.frame_stats)
        total_unripe = sum(frame['unripe_count'] for frame in self.frame_stats)
        total_unknown = sum(frame['unknown_count'] for frame in self.frame_stats)
        total_apples = sum(frame['total_count'] for frame in self.frame_stats)
        
        frames_with_detections = len([f for f in self.frame_stats if f['total_count'] > 0])
        avg_confidence = np.mean([f['avg_confidence'] for f in self.frame_stats if f['avg_confidence'] > 0])
        
        return {
            'total_frames': len(self.frame_stats),
            'frames_with_detections': frames_with_detections,
            'total_ripe_apples': total_ripe,
            'total_unripe_apples': total_unripe,
            'total_unknown_apples': total_unknown,
            'total_apples': total_apples,
            'ripe_percentage': (total_ripe / total_apples * 100) if total_apples > 0 else 0,
            'unripe_percentage': (total_unripe / total_apples * 100) if total_apples > 0 else 0,
            'unknown_percentage': (total_unknown / total_apples * 100) if total_apples > 0 else 0,
            'average_confidence': avg_confidence if not np.isnan(avg_confidence) else 0,
            'detection_rate': (frames_with_detections / len(self.frame_stats) * 100) if self.frame_stats else 0
        }

def process_video_hybrid(input_path, output_path, model_path='yolov8n.pt', 
                        confidence_threshold=0.3, sample_frames=10):
    """
    Process video with hybrid YOLO + color analysis approach
    
    Args:
        input_path: Input video path
        output_path: Output video path  
        model_path: YOLO model path
        confidence_threshold: Detection confidence threshold
        sample_frames: Process every N frames for speed (set to 1 for all frames)
    """
    
    print(f"🚀 Starting hybrid apple ripeness detection...")
    print(f"📹 Input: {input_path}")
    print(f"🎯 Output: {output_path}")
    print(f"🤖 Model: {model_path}")
    print(f"📊 Confidence threshold: {confidence_threshold}")
    
    detector = HybridAppleDetector(model_path, confidence_threshold)
    evaluator = RipenessEvaluator()
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {input_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📺 Video info: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_frames == 0:
            print(f"\n🔄 Processing frame {frame_count}/{total_frames}")
            
            processed_frame, apple_counts, detections_info = detector.detect_and_classify_apples(frame)

            evaluator.add_frame_predictions(detections_info, None, frame_count)
            
            processed_count += 1
            
            print(f"📊 Frame results: {apple_counts}")
        else:
            processed_frame = frame
        
        y_pos = 30
        cv2.putText(processed_frame, f'Frame: {frame_count}/{total_frames}', 
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if frame_count % sample_frames == 0 and 'apple_counts' in locals():
            y_pos += 40
            cv2.putText(processed_frame, f'Total: {apple_counts["total"]}', 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            y_pos += 40
            cv2.putText(processed_frame, f'Ripe: {apple_counts["ripe"]}', 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            y_pos += 40
            cv2.putText(processed_frame, f'Unripe: {apple_counts["unripe"]}', 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        out.write(processed_frame)
        
        if frame_count % 30 == 0: 
            display_frame = cv2.resize(processed_frame, (800, 600))
            cv2.imshow('Hybrid Apple Detection', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            fps_actual = frame_count / elapsed_time
            progress = (frame_count / total_frames) * 100
            print(f"⏱️  Progress: {progress:.1f}% | FPS: {fps_actual:.1f} | Processed: {processed_count} frames")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print_comprehensive_results(output_path, frame_count, processed_count, start_time, evaluator)

def print_comprehensive_results(output_path, total_frames, processed_frames, start_time, evaluator):
    """Print comprehensive results"""
    print(f"\n{'='*70}")
    print("🍎 HYBRID APPLE RIPENESS DETECTION RESULTS")
    print(f"{'='*70}")
    
    total_time = time.time() - start_time
    print(f"✅ Processing complete!")
    print(f"📁 Output saved: {output_path}")
    print(f"🎞️  Total frames: {total_frames}")
    print(f"🔄 Processed frames: {processed_frames}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"🚀 Average FPS: {total_frames / total_time:.1f}")
    
    summary_stats = evaluator.get_summary_statistics()
    if summary_stats and summary_stats['total_apples'] > 0:
        print(f"\n{'='*70}")
        print("📊 DETECTION SUMMARY")
        print(f"{'='*70}")
        print(f"🎯 Detection rate: {summary_stats['detection_rate']:.1f}% of frames")
        print(f"🍎 Total apples detected: {summary_stats['total_apples']}")
        print(f"🟢 Ripe apples: {summary_stats['total_ripe_apples']:4d} ({summary_stats['ripe_percentage']:.1f}%)")
        print(f"🔴 Unripe apples: {summary_stats['total_unripe_apples']:4d} ({summary_stats['unripe_percentage']:.1f}%)")
        print(f"❓ Unknown: {summary_stats['total_unknown_apples']:4d} ({summary_stats['unknown_percentage']:.1f}%)")
        print(f"📈 Average confidence: {summary_stats['average_confidence']:.3f}")
        
        print(f"\n🏆 PERFORMANCE ASSESSMENT:")
        if summary_stats['total_apples'] >= 50:
            print("✅ Sufficient detections for reliable analysis")
        else:
            print("⚠️  Low detection count - consider adjusting confidence threshold")
            
        if summary_stats['unknown_percentage'] < 20:
            print("✅ Good classification rate")
        else:
            print("⚠️  High unknown classification rate - may need algorithm tuning")
            
    else:
        print(f"\n{'='*70}")
        print("❌ NO APPLES DETECTED")
        print(f"{'='*70}")
        print("Possible issues:")
        print("1. 🎯 Confidence threshold too high - try lowering to 0.1-0.3")
        print("2. 📹 Video doesn't contain apples visible to YOLO")
        print("3. 🤖 Model might not recognize apples in this video style")
        print("4. 🔍 Try different YOLO model (yolov8s.pt, yolov8m.pt)")

if __name__ == "__main__":
    input_video = 'apple.mp4'
    output_video = 'output_hybrid_detection.mp4'
    
    confidence_threshold = 0.25
    sample_frames = 5
    
    print("🍎 HYBRID APPLE RIPENESS DETECTOR")
    print("Combines YOLO detection with advanced color analysis")
    print("-" * 50)
    
    process_video_hybrid(
        input_path=input_video,
        output_path=output_video,
        model_path='yolov8n.pt',
        confidence_threshold=confidence_threshold,
        sample_frames=sample_frames
    )