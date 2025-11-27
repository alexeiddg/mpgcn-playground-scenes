# Data Processing Pipeline

This directory contains the original data and processed outputs for the playground scene analysis project. Below is a description of the data processing pipeline and the contents of each directory.

## Data Flow Overview

The data processing pipeline follows these steps:

1. **Original Data**: Start with `videos.csv` containing information about all video clips
2. **Trimmed Videos**: Filter to `videos-trimmed.csv` containing only relevant clips with motion
3. **Object & Skeleton Detection**: Process videos through YOLOv11 to extract objects and skeletons (in `temp/`)
4. **Merged Data**: Combine skeleton and object data into JSON files (in `temp/merged/`)
5. **Tensor Creation**: Convert merged JSON files to tensor data in NPY format (in `npy/`)
6. **Dataset Creation**: Process NPY files into training/evaluation datasets (in `workdir/`)

## Directory Contents

### Root Files

- `videos.csv`: Original CSV file containing all video clips
- `videos-trimmed.csv`: CSV file containing only relevant clips with motion
- `playgroundROI.gpkg`: GeoPackage file with region of interest information

### Processed Directory

- `mpgcn_labels.csv`: Activity labels for the clips
- `prefiltered_scenes.csv`: Output of the scene selection process

### Temp Directory (not in GitHub)

This directory contains intermediate processing files:

- `objects/`: Object detection results from YOLOv11
- `skeletons/`: Skeleton detection results from YOLOv11
- `merged/`: JSON files combining skeleton and object data
- `videos/trimmed/`: Trimmed video clips with motion

### NPY Directory

Contains the final tensor data created by the `tensor-builder.ipynb` notebook:

- `*_data.npy`: Skeleton tensor data (shape: [T, M, J, C])
  - T: number of frames
  - M: maximum number of people (6)
  - J: number of joints (17)
  - C: channels (x, y, confidence)
- `*_object_data.npy`: Object tensor data (shape: [T, O, C])
  - T: number of frames
  - O: number of objects (static + dynamic)
  - C: channels (x, y, confidence)

## Processing Steps

### 1. From Videos to Tensors

The `tensor-builder.ipynb` notebook in `script/notebooks/` performs these steps:

1. Processes the merged JSON files from `temp/merged/`
2. Separates objects into static (fixed relative to camera) and dynamic (moving per frame)
3. Creates skeleton tensors with shape [T, M, J, C]
4. Creates object tensors with shape [T, O, C]
5. Saves the resulting tensors to the `npy/` directory

### 2. From Tensors to Training Data

The `playground_reader.py` in `src/reader/` performs these steps:

1. Reads the NPY files from the `npy/` directory
2. Creates stratified K-fold splits for cross-validation
3. Processes the tensors into training and evaluation datasets
4. Saves the resulting datasets as pickle files in the `workdir/` directory

## Usage Notes

- The NPY files in `./data/npy` are the ones that should be used by the reader
- The final pickle files in `./workdir` are created by the reader for model training
- The remaining included data is for demonstration purposes

## Notebooks

The `script/notebooks/` directory contains notebooks that demonstrate the data transformation process:

- `tensor-builder.ipynb`: Creates the NPY tensor files from merged JSON data
- Other notebooks show different aspects of the data processing pipeline