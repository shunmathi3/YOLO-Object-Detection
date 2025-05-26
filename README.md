# YOLO-Object-Detection
## What I Have Done
- Completed the YOLO loss function to properly assign predicted bounding boxes to ground truth boxes using IoU.
- Implemented localization, confidence, and classification losses following the YOLO paper formulation.
- Trained the YOLO network on a dataset, tuning hyperparameters for optimal performance.
- Tested the trained model, generating precision-recall curves and achieving an AP of over 30%.
- Created visualization tools to display bounding box detections on images.

## How to rum
1. Clone this repo: https://github.com/shunmathi3/YOLO-Object-Detection.git
2. Install dependencies: pip install -r requirement.txt
3. Ensure data in data folder contains image and annotation files
4. Run Data Loader Test: python yolo/data.py
5. Train the Model: python yolo/train.py
6. Test and Evaluate: python yolo/test.py

