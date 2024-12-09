import cv2
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

class EnhancedVideoFrameExtractor:
    def __init__(self):
        """Initialize the EnhancedVideoFrameExtractor class"""
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        
    def check_frame_quality(self, frame, min_brightness=20, min_contrast=30, 
                          min_sharpness=50, min_size=(200, 200)):
        """
        Check frame quality using multiple metrics
        
        Parameters:
        -----------
        frame : np.ndarray
            Input frame
        min_brightness : int
            Minimum average brightness threshold (0-255)
        min_contrast : int
            Minimum contrast threshold
        min_sharpness : float
            Minimum sharpness threshold
        min_size : tuple
            Minimum frame dimensions (width, height)
            
        Returns:
        --------
        bool
            True if frame passes quality checks
        """
        try:
            # Size check
            height, width = frame.shape[:2]
            if width < min_size[0] or height < min_size[1]:
                return False
            
            # Convert to grayscale for analysis
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
                
            # Brightness check
            brightness = np.mean(gray)
            if brightness < min_brightness:
                return False
                
            # Contrast check
            contrast = np.std(gray)
            if contrast < min_contrast:
                return False
                
            # Sharpness check (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharpness < min_sharpness:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error in quality check: {str(e)}")
            return False
    
    def extract_frames(self, video_path, output_dir, sample_rate=3,  # Changed from 15 to 3
                      quality_check=True, max_frames=None):
        """
        Extract frames from video with enhanced options
        
        Parameters:
        -----------
        video_path : str
            Path to input video file
        output_dir : str
            Path to output directory for frames
        sample_rate : int
            Extract every nth frame (default: 3 for ~10 fps from 30fps video)
        quality_check : bool
            Whether to perform quality checks
        max_frames : int
            Maximum number of frames to extract (None = no limit)
        """
        try:
            # Validate video format
            if not any(video_path.lower().endswith(fmt) for fmt in self.supported_formats):
                raise ValueError(f"Unsupported video format. Supported: {self.supported_formats}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Error opening video file")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps
            
            print("Video Properties:")
            print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"FPS: {fps}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Total Frames: {total_frames}")
            print(f"Expected Output FPS: {fps/sample_rate:.1f}")
            
            # Initialize counters
            processed_count = 0
            saved_count = 0
            skipped_quality = 0
            
            # Calculate total frames to process
            frames_to_process = min(total_frames, max_frames) if max_frames else total_frames
            pbar = tqdm(total=frames_to_process//sample_rate, desc="Extracting Frames")
            
            # Initialize frame change detection
            min_frame_diff = 0.05  # Reduced threshold for more sensitivity
            prev_frame = None
            
            while cap.isOpened() and (not max_frames or processed_count < max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if processed_count % sample_rate == 0:
                    save_frame = True
                    
                    if quality_check:
                        # Basic quality check
                        save_frame = self.check_frame_quality(
                            frame,
                            min_brightness=20,  # Reduced for darker scenes
                            min_contrast=25,    # Reduced for underwater scenes
                            min_sharpness=45    # Reduced for water motion
                        )
                        
                        # Frame difference check (more lenient)
                        if save_frame and prev_frame is not None:
                            diff = np.mean(cv2.absdiff(frame, prev_frame))
                            if diff < min_frame_diff:
                                save_frame = False
                    
                    if save_frame:
                        # Generate unique filename with frame number for easier sequencing
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"frame_{processed_count:06d}_{timestamp}.jpg"
                        output_path = os.path.join(output_dir, filename)
                        
                        # Save frame
                        cv2.imwrite(output_path, frame)
                        saved_count += 1
                        prev_frame = frame.copy()
                    else:
                        skipped_quality += 1
                    
                    pbar.update(1)
                
                processed_count += 1
            
            pbar.close()
            cap.release()
            
            # Print statistics
            print(f"\nExtraction Complete:")
            print(f"Total frames processed: {processed_count}")
            print(f"Frames saved: {saved_count}")
            print(f"Effective FPS: {saved_count/duration:.1f}")
            print(f"Frames skipped (quality): {skipped_quality}")
            print(f"Frames skipped (sampling): {processed_count - (processed_count//sample_rate)}")
            
            return saved_count > 0
            
        except Exception as e:
            print(f"Error during frame extraction: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def get_video_metadata(self, video_path):
        """
        Get detailed video metadata
        
        Parameters:
        -----------
        video_path : str
            Path to video file
            
        Returns:
        --------
        dict
            Dictionary containing video metadata
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Error opening video file")
            
            metadata = {
                'filename': os.path.basename(video_path),
                'format': os.path.splitext(video_path)[1],
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': int(cap.get(cv2.CAP_PROP_FPS)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 
                              cap.get(cv2.CAP_PROP_FPS)),
                'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
                'file_size_mb': os.path.getsize(video_path) / (1024 * 1024)
            }
            
            cap.release()
            return metadata
            
        except Exception as e:
            print(f"Error getting video metadata: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    try:
        extractor = EnhancedVideoFrameExtractor()
        
        # Paths
        VIDEO_PATH = 'C:/Users/ssang/OneDrive/Documents/Python repo/Sem2/CompAI/Project/Breast_Stroke.mp4'
        OUTPUT_DIR = 'C:/Users/ssang/OneDrive/Documents/Python repo/Sem2/CompAI/Project/Breast_Stroke_extracted_frames'
        
        # Get video metadata
        metadata = extractor.get_video_metadata(VIDEO_PATH)
        if metadata:
            print("\nVideo Metadata:")
            for key, value in metadata.items():
                print(f"{key}: {value}")
        
        # Extract frames with higher frequency
        extractor.extract_frames(
            video_path=VIDEO_PATH,
            output_dir=OUTPUT_DIR,
            sample_rate=3,  # Extract every 3rd frame (10 fps from 30fps video)
            quality_check=True,
            max_frames=None  # No limit
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()