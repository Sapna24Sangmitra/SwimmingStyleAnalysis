import cv2
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json

@dataclass
class StyleClassifierConfig:
    """Configuration for swimming style classification"""
    sequence_length: int = 5
    confidence_threshold: float = 0.5
    temporal_smoothing: bool = True
    smooth_window: int = 3
    min_stick_points: int = 8  # Minimum number of points needed in stick figure

class SwimmingStyleClassifier:
    def __init__(self, config: StyleClassifierConfig = None):
        """Initialize Swimming Style Classifier"""
        self.config = config or StyleClassifierConfig()
        
        # Define keypoint indices for stick figure analysis
        self.keypoints = {
            'nose': 0,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }
        
        self.style_history = []

    def extract_keypoints_from_image(self, image_path: str) -> np.ndarray:
        """
        Extract keypoints from stick figure image
        
        Parameters:
        -----------
        image_path : str
            Path to stick figure image
            
        Returns:
        --------
        np.ndarray
            Array of keypoint coordinates
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find all red points (joints in stick figure)
            red_mask = cv2.inRange(image, (0, 0, 200), (50, 50, 255))
            keypoints = cv2.findNonZero(red_mask)
            
            if keypoints is None or len(keypoints) < self.config.min_stick_points:
                return None
            
            # Sort points by y-coordinate first, then x-coordinate
            keypoints = sorted(keypoints, key=lambda x: (x[0][1], x[0][0]))
            
            # Convert to numpy array
            keypoints_array = np.array([[kp[0][0], kp[0][1]] for kp in keypoints])
            
            return keypoints_array
            
        except Exception as e:
            print(f"Error extracting keypoints from {image_path}: {str(e)}")
            return None

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        if not all(p is not None for p in [p1, p2, p3]):
            return None
            
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def extract_pose_features(self, keypoints: np.ndarray) -> Dict:
        """Extract relevant features from keypoints"""
        features = {}
        
        try:
            if len(keypoints) < self.config.min_stick_points:
                return None
            
            # Estimate key points from sorted keypoints
            points = {}
            points['nose'] = keypoints[0]  # Top-most point
            
            # Shoulders (next two points)
            points['left_shoulder'] = keypoints[1]
            points['right_shoulder'] = keypoints[2]
            
            # Elbows
            points['left_elbow'] = keypoints[3]
            points['right_elbow'] = keypoints[4]
            
            # Wrists
            points['left_wrist'] = keypoints[5]
            points['right_wrist'] = keypoints[6]
            
            # Hips
            mid_idx = len(keypoints) // 2
            points['left_hip'] = keypoints[mid_idx - 1]
            points['right_hip'] = keypoints[mid_idx]
            
            # Knees and ankles
            points['left_knee'] = keypoints[-4]
            points['right_knee'] = keypoints[-3]
            points['left_ankle'] = keypoints[-2]
            points['right_ankle'] = keypoints[-1]
            
            # Calculate features
            # Body orientation
            spine_vector = (points['left_shoulder'] + points['right_shoulder']) / 2
            hip_center = (points['left_hip'] + points['right_hip']) / 2
            features['body_angle'] = np.degrees(np.arctan2(
                spine_vector[1] - hip_center[1],
                spine_vector[0] - hip_center[0]
            ))
            
            # Arm angles
            features['left_arm_angle'] = self.calculate_angle(
                points['left_wrist'],
                points['left_elbow'],
                points['left_shoulder']
            )
            
            features['right_arm_angle'] = self.calculate_angle(
                points['right_wrist'],
                points['right_elbow'],
                points['right_shoulder']
            )
            
            # Leg angles
            features['left_leg_angle'] = self.calculate_angle(
                points['left_ankle'],
                points['left_knee'],
                points['left_hip']
            )
            
            features['right_leg_angle'] = self.calculate_angle(
                points['right_ankle'],
                points['right_knee'],
                points['right_hip']
            )
            
            # Symmetry
            features['arm_symmetry'] = abs(features['left_arm_angle'] - features['right_arm_angle'])
            features['leg_symmetry'] = abs(features['left_leg_angle'] - features['right_leg_angle'])
            
            # Height differences
            features['arms_height'] = (
                (points['left_wrist'][1] + points['right_wrist'][1]) / 2 -
                (points['left_shoulder'][1] + points['right_shoulder'][1]) / 2
            )
            
            features['legs_height'] = (
                (points['left_ankle'][1] + points['right_ankle'][1]) / 2 -
                (points['left_hip'][1] + points['right_hip'][1]) / 2
            )
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def classify_style(self, features: Dict) -> Tuple[str, float]:
        """Classify swimming style based on pose features"""
        try:
            if not features:
                return 'unknown', 0.0
            
            # Confidence thresholds
            high_conf = 0.8
            medium_conf = 0.6
            
            # Butterfly detection
            if (abs(features['body_angle']) < 30 and
                features['arm_symmetry'] < 20 and
                features['leg_symmetry'] < 20 and
                abs(features['arms_height']) > 50):
                return 'butterfly', high_conf
            
            # Backstroke detection
            if (abs(features['body_angle']) > 150 and
                features['arm_symmetry'] > 45):
                return 'backstroke', high_conf
            
            # Breaststroke detection
            if (abs(features['body_angle']) < 45 and
                features['arm_symmetry'] < 25 and
                features['leg_symmetry'] < 25 and
                features['legs_height'] < -30):
                return 'breaststroke', high_conf
            
            # Freestyle detection
            if (abs(features['body_angle']) < 45 and
                features['arm_symmetry'] > 45 and
                features['legs_height'] > -50):
                return 'freestyle', medium_conf
            
            return 'unknown', 0.3
            
        except Exception as e:
            print(f"Error in style classification: {str(e)}")
            return 'unknown', 0.0

    def smooth_style_prediction(self, style: str, confidence: float) -> Tuple[str, float]:
        """Apply temporal smoothing to style predictions"""
        if not self.config.temporal_smoothing:
            return style, confidence
        
        self.style_history.append((style, confidence))
        
        if len(self.style_history) > self.config.smooth_window:
            self.style_history.pop(0)
        
        style_counts = {}
        total_confidence = {}
        
        for s, c in self.style_history:
            style_counts[s] = style_counts.get(s, 0) + 1
            total_confidence[s] = total_confidence.get(s, 0) + c
        
        max_count = 0
        smoothed_style = style
        smoothed_confidence = confidence
        
        for s, count in style_counts.items():
            if count > max_count:
                max_count = count
                smoothed_style = s
                smoothed_confidence = total_confidence[s] / count
        
        return smoothed_style, smoothed_confidence

    def process_stick_figures(self, input_dir: str, output_dir: str) -> None:
        """
        Process folder of stick figure images
        
        Parameters:
        -----------
        input_dir : str
            Path to directory containing stick figure images
        output_dir : str
            Path to output directory
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get all images
            image_files = [f for f in os.listdir(input_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                print("No images found in input directory")
                return
            
            # Sort files by name (assuming sequential naming)
            image_files.sort()
            
            # Process each image
            classifications = []
            print("\nClassifying swimming styles...")
            
            for image_file in tqdm(image_files):
                image_path = os.path.join(input_dir, image_file)
                
                # Extract keypoints
                keypoints = self.extract_keypoints_from_image(image_path)
                
                if keypoints is not None:
                    # Extract features
                    features = self.extract_pose_features(keypoints)
                    
                    if features:
                        # Classify style
                        style, confidence = self.classify_style(features)
                        
                        # Apply temporal smoothing
                        smoothed_style, smoothed_confidence = self.smooth_style_prediction(
                            style, confidence
                        )
                        
                        # Store results
                        classifications.append({
                            'filename': image_file,
                            'style': smoothed_style,
                            'confidence': float(smoothed_confidence),
                            'features': {k: float(v) if isinstance(v, (int, float)) else v 
                                       for k, v in features.items()}
                        })
            
            # Analyze sequence
            styles = [c['style'] for c in classifications if c['style'] != 'unknown']
            if styles:
                unique_styles = set(styles)
                style_counts = {s: styles.count(s) for s in unique_styles}
                primary_style = max(style_counts.items(), key=lambda x: x[1])[0]
                primary_confidence = style_counts[primary_style] / len(styles)
                
                print(f"\nSequence Analysis:")
                print(f"Primary swimming style: {primary_style}")
                print(f"Confidence: {primary_confidence:.2f}")
                print("\nStyle distribution:")
                for style, count in style_counts.items():
                    percentage = count / len(styles) * 100
                    print(f"{style}: {count} frames ({percentage:.1f}%)")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"style_classification_{timestamp}.json")
            
            with open(output_path, 'w') as f:
                json.dump({
                    'frame_classifications': classifications,
                    'sequence_summary': {
                        'primary_style': primary_style,
                        'confidence': primary_confidence,
                        'style_distribution': style_counts
                    }
                }, f, indent=4)
            
            print(f"\nResults saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing stick figures: {str(e)}")
            import traceback
            traceback.print_exc()

# Example usage
if __name__ == "__main__":
    try:
        # Hardcoded paths - modify these according to your directory structure
        INPUT_DIR = 'C://Users//ssang//OneDrive//Documents//Python repo//Sem2//CompAI//Project//Breast_Stroke_yolo_estimation'
        OUTPUT_DIR = 'C://Users//ssang//OneDrive//Documents//Python repo//Sem2//CompAI//Project//Breast_Stroke_style_classification'
        
        # Configure classifier
        config = StyleClassifierConfig(
            sequence_length=5,
            confidence_threshold=0.5,
            temporal_smoothing=True,
            smooth_window=3,
            min_stick_points=8
        )
        
        # Initialize classifier
        classifier = SwimmingStyleClassifier(config)
        
        # Process stick figures
        print("\nStarting swimming style classification...")
        print(f"Input directory: {INPUT_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        
        classifier.process_stick_figures(INPUT_DIR, OUTPUT_DIR)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nProcess complete")