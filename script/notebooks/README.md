# MPGCN Playground Notebooks

This directory contains Jupyter notebooks for the MPGCN (Multi-Person Graph Convolutional Network) playground project. These notebooks cover the entire pipeline from data acquisition to model training and evaluation.

## Notebook Overview and Logical Order

Follow these notebooks in sequence to replicate the complete workflow:

> Please note the notebooks have been moved to organized folders, make sure the paths like './../../data/npy' match to their new location.

### 1. Data Acquisition and Preparation

1. **get-azure-blobs.ipynb**
   - Connects to Azure Blob Storage
   - Lists and validates available video files

2. **get-raw-videos.ipynb**
   - Downloads raw videos from Azure Blob Storage
   - Trims videos to specific activity windows using ffmpeg
   - Creates a dataframe with information about the trimmed videos

3. **get-trimmed-videos.ipynb**
   - Processes the trimmed videos for further analysis
   - Prepares videos for object detection and skeleton tracking

### 2. Feature Extraction

4. **object-detection.ipynb**
   - Uses YOLO model for object detection on trimmed videos
   - Extracts bounding boxes, confidence scores, and class labels
   - Saves detection results as JSON files

5. **skeleton-tracking.ipynb**
   - Uses YOLO pose estimation to detect human keypoints
   - Applies DeepSort for tracking people across frames
   - Extracts bounding boxes and keypoints for each person
   - Saves tracking results as JSON files

6. **merge-pose-objs.ipynb**
   - Merges results from pose estimation and object detection
   - Filters objects to include only relevant classes
   - Creates a unified representation of people and objects
   - Saves merged data as JSON files

### 3. Data Processing and Tensor Creation

7. **tensor-builder.ipynb**
   - Separates objects into static and dynamic categories
   - Extracts camera-specific static object information
   - Analyzes the distribution of people per frame
   - Builds tensors for skeleton and object data
   - Saves tensors as NPY files for model training

8. **labeling-videos.ipynb**
   - Creates labels for the video clips
   - Associates each clip with its corresponding activity class

### 4. Model Training and Evaluation

9. **splits_test.ipynb**
   - Tests and analyzes train/test splits
   - Verifies class distribution in train and evaluation sets
   - Checks camera distribution in the datasets
   - Creates and validates dataset objects for training

10. **overfit_test.ipynb**
    - Tests the model's ability to overfit on a single sample
    - Verifies that the model can learn and has sufficient capacity
    - Serves as a debugging tool for model implementation

11. **mpgcn_diagnostics.ipynb**
    - Provides comprehensive diagnostics of the MPGCN model
    - Visualizes the graph structure used by the model
    - Demonstrates data augmentation techniques
    - Analyzes model outputs and features

### 5. Visualization and Analysis

12. **skeleton-visualization.ipynb**
    - Visualizes the extracted skeleton data
    - Helps verify the quality of pose estimation

13. **scene-selection.ipynb**
    - Tools for selecting and analyzing specific scenes
    - Helps curate the dataset for better model performance

## Additional Notebooks

- **gendata_test.ipynb**: Tests the data generation process
- **mpgcn_playground.ipynb**: Main notebook for experimenting with the MPGCN model

## Getting Started

To replicate the complete workflow:

1. Start with the data acquisition notebooks (1-3)
2. Proceed to feature extraction (4-6)
3. Process the data and create tensors (7-8)
4. Train and evaluate the model (9-11)
5. Use the visualization notebooks (12-14) as needed for analysis

Each notebook contains detailed comments explaining the steps and parameters used.

## Requirements

The notebooks require various dependencies including:
- PyTorch
- YOLO (Ultralytics)
- DeepSort
- NumPy
- Pandas
- Matplotlib
- NetworkX (for graph visualization)

Make sure to install all required dependencies before running the notebooks.
