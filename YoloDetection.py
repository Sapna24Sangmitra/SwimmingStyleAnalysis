import torch
import cv2
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO
from collections import deque

class EnhancedSwimmerDetection:
    def __init__(self, model_path='yolov8x.pt', confidence_threshold=0.6):
        """
        Initialize Enhanced Swimmer Detection system
        
        Parameters:
        -----------
        model_path : str
            Path to YOLO model weights
        confidence_threshold : float
            Detection confidence threshold
        """
        # Load YOLO model with best weights
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0  # COCO dataset person class
        
        # Configure model
        self.model.conf = confidence_threshold
        self.model.classes = [self.person_class_id]
        
        # Pool detection parameters
        self.pool_color_ranges = [
            # Light blue to dark blue
            {
                'lower': np.array([90, 50, 50]),
                'upper': np.array([130, 255, 255])
            },
            # Turquoise range
            {
                'lower': np.array([80, 40, 40]),
                'upper': np.array([100, 255, 255])
            }
        ]
        
        # Detection parameters
        self.detection_params = {
            'min_aspect_ratio': 0.3,
            'max_aspect_ratio': 3.0,
            'min_area_ratio': 0.01,
            'max_area_ratio': 0.4,
            'min_pool_overlap': 0.3,
            'max_vertical_ratio': 2.5,
            'min_detection_size': (30, 30)
        }
        
        # Tracking parameters
        self.track_history = {}
        self.max_track_length = 30
        self.min_track_length = 3
        self.max_track_gap = 10
        
    def detect_pool_area(self, image):
        """
        Enhanced pool area detection using multiple color ranges and morphology
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
            
        Returns:
        --------
        np.ndarray
            Binary mask of detected pool area
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Try each color range
        for color_range in self.pool_color_ranges:
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Advanced morphological operations
        kernel_small = np.ones((5,5), np.uint8)
        kernel_large = np.ones((20,20), np.uint8)
        
        # Remove noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        # Fill holes
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large)
        
        # Find largest contour (assumed to be the pool)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(combined_mask)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            return mask
        
        return combined_mask
    
    def validate_detection(self, image, bbox, pool_mask):
        """
        Enhanced detection validation with multiple criteria
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
        bbox : tuple
            Bounding box coordinates (x1, y1, x2, y2)
        pool_mask : np.ndarray
            Binary mask of pool area
            
        Returns:
        --------
        bool
            True if detection is valid
        """
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        
        # Size check
        if width < self.detection_params['min_detection_size'][0] or \
           height < self.detection_params['min_detection_size'][1]:
            return False
        
        # Aspect ratio check
        aspect_ratio = height / width
        if not (self.detection_params['min_aspect_ratio'] <= aspect_ratio <= 
                self.detection_params['max_aspect_ratio']):
            return False
        
        # Area ratio check
        detection_area = height * width
        image_area = image.shape[0] * image.shape[1]
        area_ratio = detection_area / image_area
        if not (self.detection_params['min_area_ratio'] <= area_ratio <= 
                self.detection_params['max_area_ratio']):
            return False
        
        # Pool overlap check
        detection_mask = np.zeros_like(pool_mask)
        detection_mask[y1:y2, x1:x2] = 1
        overlap = np.logical_and(pool_mask, detection_mask)
        overlap_ratio = np.sum(overlap) / np.sum(detection_mask)
        if overlap_ratio < self.detection_params['min_pool_overlap']:
            return False
        
        # Vertical ratio check (avoid poles/lane markers)
        if height / width > self.detection_params['max_vertical_ratio']:
            return False
        
        return True
    
    def update_tracks(self, detections, frame_id):
        """
        Update detection tracks for temporal consistency
        
        Parameters:
        -----------
        detections : list
            List of current detections
        frame_id : int
            Current frame ID
        
        Returns:
        --------
        list
            Filtered detections
        """
        # Convert detections to centroids
        current_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            current_centroids.append((centroid, det))
        
        # Update existing tracks
        active_tracks = {}
        
        for track_id, track_info in self.track_history.items():
            last_frame = track_info['frames'][-1]
            if frame_id - last_frame <= self.max_track_gap:
                active_tracks[track_id] = track_info
        
        # Match detections to tracks
        matched_detections = set()
        for track_id, track_info in active_tracks.items():
            if current_centroids:
                last_centroid = track_info['centroids'][-1]
                # Find closest detection
                min_dist = float('inf')
                best_match = None
                
                for i, (centroid, det) in enumerate(current_centroids):
                    if i not in matched_detections:
                        dist = np.sqrt((centroid[0] - last_centroid[0])**2 + 
                                     (centroid[1] - last_centroid[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_match = i
                
                if best_match is not None and min_dist < 100:  # Max distance threshold
                    matched_detections.add(best_match)
                    track_info['centroids'].append(current_centroids[best_match][0])
                    track_info['frames'].append(frame_id)
                    if len(track_info['centroids']) > self.max_track_length:
                        track_info['centroids'].pop(0)
                        track_info['frames'].pop(0)
        
        # Create new tracks for unmatched detections
        for i, (centroid, det) in enumerate(current_centroids):
            if i not in matched_detections:
                track_id = len(self.track_history)
                self.track_history[track_id] = {
                    'centroids': deque([centroid], maxlen=self.max_track_length),
                    'frames': deque([frame_id], maxlen=self.max_track_length)
                }
        
        # Filter detections based on track history
        filtered_detections = []
        for i, (centroid, det) in enumerate(current_centroids):
            # Find corresponding track
            for track_info in self.track_history.values():
                if centroid == track_info['centroids'][-1]:
                    if len(track_info['frames']) >= self.min_track_length:
                        filtered_detections.append(det)
                    break
        
        return filtered_detections
    
    def detect_swimmers(self, image, frame_id=0):
        """
        Detect swimmers with enhanced filtering and tracking
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
        frame_id : int
            Current frame ID for tracking
            
        Returns:
        --------
        tuple
            (detections, pool_mask)
        """
        # Detect pool area
        pool_mask = self.detect_pool_area(image)
        
        # Run YOLO detection
        results = self.model(image)
        
        # Filter detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = box.cls[0].cpu().numpy()
                
                if (class_id == self.person_class_id and 
                    confidence >= self.confidence_threshold):
                    
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    if self.validate_detection(image, bbox, pool_mask):
                        detections.append({
                            'bbox': bbox,
                            'confidence': float(confidence)
                        })
        
        # Apply temporal filtering
        filtered_detections = self.update_tracks(detections, frame_id)
        
        return filtered_detections, pool_mask

    def process_image(self, image_path, output_dir, frame_id=0):
        """
        Process single image with visualization
        
        Parameters:
        -----------
        image_path : str
            Path to input image
        output_dir : str
            Path to output directory
        frame_id : int
            Current frame ID for tracking
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Detect swimmers
            detections, pool_mask = self.detect_swimmers(image, frame_id)
            
            if not detections:
                return False
            
            # Generate output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Create visualization
            result_image = image.copy()
            
            # Draw pool area
            pool_overlay = cv2.cvtColor(pool_mask, cv2.COLOR_GRAY2BGR)
            pool_overlay[pool_mask > 0] = [100, 100, 0]
            cv2.addWeighted(result_image, 0.7, pool_overlay, 0.3, 0, result_image)
            
            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                # Draw box with thickness based on confidence
                thickness = max(1, int(det['confidence'] * 3))
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                
                # Add confidence label
                label = f"Swimmer {det['confidence']:.2f}"
                cv2.putText(result_image, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save results
            output_path = os.path.join(output_dir, 
                                     f"detection_{timestamp}_{base_name}.jpg")
            cv2.imwrite(output_path, result_image)
            
            # Save individual crops with padding
            padding = 10
            height, width = image.shape[:2]
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                # Add padding
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(width, x2 + padding)
                y2 = min(height, y2 + padding)
                
                crop = image[y1:y2, x1:x2]
                crop_path = os.path.join(output_dir,
                                       f"swimmer_{timestamp}_{base_name}_{i}.jpg")
                cv2.imwrite(crop_path, crop)
            
            return True
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def batch_process(self, input_dir, output_dir):
        """
        Process multiple images
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            image_files = [f for f in os.listdir(input_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                print("No images found in input directory")
                return
            
            swimmers_found = 0
            errors = 0
            
            print(f"\nProcessing {len(image_files)} images...")
            for frame_id, image_file in enumerate(tqdm(image_files)):
                try:
                    image_path = os.path.join(input_dir, image_file)
                    if self.process_image(image_path, output_dir, frame_id):
                        swimmers_found += 1
                except Exception as e:
                    errors += 1
                    print(f"\nError processing {image_file}: {str(e)}")
                    continue
            
            print(f"\nProcessing Complete:")
            print(f"Total images processed: {len(image_files)}")
            print(f"Images with swimmers: {swimmers_found}")
            print(f"Errors encountered: {errors}")
            
        except Exception as e:
            print(f"Error during batch processing: {str(e)}")
            import traceback
            traceback.print_exc()

# Example usage
if __name__ == "__main__":
    try:
        detector = EnhancedSwimmerDetection(
            model_path='yolov8x.pt',
            confidence_threshold=0.6
        )
        
        INPUT_DIR = 'C://Users//ssang//OneDrive//Documents//Python repo//Sem2//CompAI//Project//Breast_Stroke_extracted_frames'
        OUTPUT_DIR = 'C://Users//ssang//OneDrive//Documents//Python repo//Sem2//CompAI//Project//Breast_Stroke_yolo_detections'
        
        detector.batch_process(INPUT_DIR, OUTPUT_DIR)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()