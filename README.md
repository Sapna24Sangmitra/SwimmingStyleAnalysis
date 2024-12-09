# Swimming Style Analysis Using YOLOv8

This project implements an automated swimming style analysis system using computer vision and deep learning techniques. The system processes video footage of swimmers to classify four main swimming styles: breaststroke, backstroke, butterfly, and freestyle.

## Overview

The system consists of four main components:
- Frame Extraction Module
- Swimmer Detection Module
- Pose Estimation Module
- Style Classification Module

## Features

- Video frame extraction with quality assessment
- Swimmer detection using YOLOv8
- Pose estimation using YOLOv8-pose
- Swimming style classification
- Temporal smoothing for stable predictions
- Support for multiple video formats (.mp4, .avi, .mov, .mkv)

## Prerequisites

```
- Python 3.8 or higher
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- NumPy
- tqdm
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/swimming-style-analysis.git
cd swimming-style-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 weights:
```bash
# For detection
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

# For pose estimation
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt
```

## Usage

### Frame Extraction

```python
from frame_extractor import EnhancedVideoFrameExtractor

extractor = EnhancedVideoFrameExtractor()
extractor.extract_frames(
    video_path='path/to/video.mp4',
    output_dir='path/to/output',
    sample_rate=3
)
```

### Swimmer Detection

```python
from yolo_detection import EnhancedSwimmerDetection

detector = EnhancedSwimmerDetection(
    model_path='yolov8x.pt',
    confidence_threshold=0.6
)
detector.process_image('path/to/image.jpg', 'path/to/output')
```

### Pose Estimation

```python
from yolo_estimation import SwimmerYoloPoseEstimation

pose_estimator = SwimmerYoloPoseEstimation()
pose_estimator.process_image('path/to/image.jpg', 'path/to/output')
```

### Style Classification

```python
from style_classification import SwimmingStyleClassifier

classifier = SwimmingStyleClassifier()
classifier.process_stick_figures('path/to/poses', 'path/to/output')
```

## Project Structure

```
swimming-style-analysis/
├── frame_extractor.py      # Frame extraction module
├── yolo_detection.py       # Swimmer detection module
├── yolo_estimation.py      # Pose estimation module
├── style_classification.py # Style classification module
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Configuration

Key parameters can be adjusted in each module:

### Frame Extraction
- `sample_rate`: Frame sampling rate (default: 3)
- `min_brightness`: Minimum frame brightness (default: 20)
- `min_contrast`: Minimum frame contrast (default: 30)

### Swimmer Detection
- `confidence_threshold`: Detection confidence threshold (default: 0.6)
- `iou_threshold`: IOU threshold for NMS (default: 0.7)

### Pose Estimation
- `conf_threshold`: Pose confidence threshold (default: 0.5)
- `smooth_window`: Temporal smoothing window (default: 5)

### Style Classification
- `sequence_length`: Frames for analysis (default: 5)
- `smooth_window`: Classification smoothing window (default: 3)

## Output

The system generates:
- Extracted video frames
- Swimmer detection results with bounding boxes
- Pose estimation visualizations
- Style classification results in JSON format

## Known Limitations

- Frame extraction rate limited to 10 FPS from 30 FPS videos
- Real-time processing not currently supported
- GPU recommended for optimal performance
- Requires clear pool visibility for best results

## Future Improvements

- Adaptive sampling implementation
- Real-time processing optimization
- Enhanced analysis features
