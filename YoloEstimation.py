import cv2
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class YoloPoseConfig:
    """Configuration for YOLO pose estimation"""
    model_path: str = 'yolov8x-pose.pt'  # Using the largest pose model for best accuracy
    conf_threshold: float = 0.5
    iou_threshold: float = 0.7
    device: str = 'cuda'  # Use 'cpu' if no GPU available
    smooth_landmarks: bool = True
    smooth_window: int = 5

class SwimmerYoloPoseEstimation:
    def __init__(self, config: YoloPoseConfig = None):
        """
        Initialize YOLO Pose Estimation for Swimming
        
        Parameters:
        -----------
        config : YoloPoseConfig
            Configuration parameters
        """
        self.config = config or YoloPoseConfig()
        
        # Load YOLO pose model
        self.model = YOLO(self.config.model_path)
        
        # Configure model parameters
        self.model.conf = self.config.conf_threshold
        self.model.iou = self.config.iou_threshold
        
        # Initialize landmark smoothing
        self.landmark_history = {}
        
        # Define keypoint connections for visualization
        self.pose_connections = [
            # Torso
            (5, 6),   # Left shoulder - Right shoulder
            (5, 11),  # Left shoulder - Left hip
            (6, 12),  # Right shoulder - Right hip
            (11, 12), # Left hip - Right hip
            
            # Arms
            (5, 7),   # Left shoulder - Left elbow
            (7, 9),   # Left elbow - Left wrist
            (6, 8),   # Right shoulder - Right elbow
            (8, 10),  # Right elbow - Right wrist
            
            # Legs
            (11, 13), # Left hip - Left knee
            (13, 15), # Left knee - Left ankle
            (12, 14), # Right hip - Right knee
            (14, 16)  # Right knee - Right ankle
        ]

    def smooth_keypoints(self, keypoints: np.ndarray, frame_id: int) -> np.ndarray:
        """
        Apply temporal smoothing to keypoints
        
        Parameters:
        -----------
        keypoints : np.ndarray
            Current frame keypoints
        frame_id : int
            Current frame ID
        
        Returns:
        --------
        np.ndarray
            Smoothed keypoints
        """
        if not self.config.smooth_landmarks:
            return keypoints
            
        # Initialize history for new frame
        if frame_id not in self.landmark_history:
            self.landmark_history[frame_id] = keypoints
            return keypoints
        
        # Get history
        history = []
        for i in range(max(0, frame_id - self.config.smooth_window), frame_id + 1):
            if i in self.landmark_history:
                history.append(self.landmark_history[i])
        
        # Apply exponential smoothing
        weights = np.exp(np.linspace(-1, 0, len(history)))
        weights /= weights.sum()
        
        smoothed = np.zeros_like(keypoints)
        for i in range(len(history)):
            smoothed += history[i] * weights[i]
        
        # Update history
        self.landmark_history[frame_id] = smoothed
        
        # Clear old history
        old_frames = [f for f in self.landmark_history 
                     if f < frame_id - self.config.smooth_window]
        for f in old_frames:
            del self.landmark_history[f]
        
        return smoothed

    def draw_pose(self, image: np.ndarray, keypoints: np.ndarray, 
                 confidence: float, style: str = 'full') -> np.ndarray:
        """
        Draw pose keypoints and connections
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
        keypoints : np.ndarray
            Pose keypoints
        confidence : float
            Detection confidence
        style : str
            Visualization style ('full', 'stick', 'points')
        
        Returns:
        --------
        np.ndarray
            Annotated image
        """
        annotated = image.copy()
        
        if style == 'full' or style == 'stick':
            # Draw connections
            for connection in self.pose_connections:
                pt1 = tuple(map(int, keypoints[connection[0]][:2]))
                pt2 = tuple(map(int, keypoints[connection[1]][:2]))
                
                # Check if points are valid
                if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                    0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
                    cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)
        
        if style == 'full' or style == 'points':
            # Draw keypoints
            for x, y, conf in keypoints:
                if conf > self.config.conf_threshold:
                    x, y = int(x), int(y)
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
        
        # Add confidence score
        cv2.putText(annotated, f"Conf: {confidence:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated

    def process_image(self, image_path: str, output_dir: str, frame_id: int = 0) -> Tuple[bool, Dict]:
        """
        Process single image for pose estimation
        
        Parameters:
        -----------
        image_path : str
            Path to input image
        output_dir : str
            Path to output directory
        frame_id : int
            Current frame ID
            
        Returns:
        --------
        Tuple[bool, Dict]
            Success flag and pose data
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return False, None
            
            # Run YOLO pose estimation
            results = self.model(image, verbose=False)
            
            if not results or len(results[0].keypoints) == 0:
                return False, None
            
            # Get keypoints and confidence
            keypoints = results[0].keypoints[0].data[0].cpu().numpy()  # Get first person
            confidence = float(results[0].boxes[0].conf)
            
            # Apply smoothing
            smoothed_keypoints = self.smooth_keypoints(keypoints, frame_id)
            
            # Create visualizations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            for style in ['full', 'stick', 'points']:
                vis_image = self.draw_pose(image, smoothed_keypoints, confidence, style)
                output_path = os.path.join(
                    output_dir,
                    f"pose_{style}_{timestamp}_{base_name}.jpg"
                )
                cv2.imwrite(output_path, vis_image)
            
            # Prepare pose data
            pose_data = {
                'keypoints': smoothed_keypoints.tolist(),
                'confidence': confidence,
                'frame_id': frame_id
            }
            
            return True, pose_data
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None

    def batch_process(self, input_dir: str, output_dir: str) -> None:
        """
        Process multiple images
        
        Parameters:
        -----------
        input_dir : str
            Path to input directory
        output_dir : str
            Path to output directory
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            image_files = [f for f in os.listdir(input_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                print("No images found in input directory")
                return
            
            pose_data_collection = []
            poses_detected = 0
            errors = 0
            
            print(f"\nProcessing {len(image_files)} images...")
            for frame_id, image_file in enumerate(tqdm(image_files)):
                try:
                    image_path = os.path.join(input_dir, image_file)
                    success, pose_data = self.process_image(image_path, output_dir, frame_id)
                    
                    if success:
                        poses_detected += 1
                        pose_data_collection.append({
                            'filename': image_file,
                            'pose_data': pose_data
                        })
                except Exception as e:
                    errors += 1
                    print(f"\nError processing {image_file}: {str(e)}")
                    continue
            
            # Save pose data
            if pose_data_collection:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                data_path = os.path.join(output_dir, f"pose_data_{timestamp}.npy")
                np.save(data_path, pose_data_collection)
            
            print(f"\nProcessing Complete:")
            print(f"Total images processed: {len(image_files)}")
            print(f"Poses detected: {poses_detected}")
            print(f"Detection rate: {poses_detected/len(image_files)*100:.1f}%")
            print(f"Errors encountered: {errors}")
            
        except Exception as e:
            print(f"Error during batch processing: {str(e)}")
            import traceback
            traceback.print_exc()

# Example usage
if __name__ == "__main__":
    try:
        # Hardcoded paths - modify these according to your directory structure
        INPUT_DIR = 'C://Users//ssang//OneDrive//Documents//Python repo//Sem2//CompAI//Project//Breast_Stroke_yolo_detections'
        OUTPUT_DIR = 'C://Users//ssang//OneDrive//Documents//Python repo//Sem2//CompAI//Project//Breast_Stroke_yolo_estimation'
        
        # Configure pose estimation
        config = YoloPoseConfig(
            model_path='yolov8x-pose.pt',
            conf_threshold=0.5,
            iou_threshold=0.7,
            device='cuda',  # Use 'cpu' if no GPU available
            smooth_landmarks=True,
            smooth_window=5
        )
        
        # Initialize pose estimator
        pose_estimator = SwimmerYoloPoseEstimation(config)
        
        # Process images
        print("\nStarting YOLO pose estimation...")
        print(f"Input directory: {INPUT_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        
        pose_estimator.batch_process(INPUT_DIR, OUTPUT_DIR)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nProcess complete")