# EuroSAT Satellite Image Classification

## Project Overview
This project implements a comprehensive computer vision pipeline for classifying satellite imagery from the EuroSAT dataset using deep learning techniques. The system can process both RGB and multispectral satellite data to classify land use and land cover into different categories (Annual Crop, Industrial, River, etc.). The implementation follows machine learning best practices with stratified data splitting, custom dataset classes, model fine-tuning, and thorough evaluation.

## Dataset
The EuroSAT dataset consists of:
- **RGB Images**: 64x64 RGB satellite images (~91MB)
- **Multispectral Images**: 13-band Sentinel-2 satellite imagery (~2.0GB)

The dataset contains 10 classes representing different land use and land cover types.

## Project Structure
```
EuroSAT Classification Project/
├── data_splitting.py         # Data preparation and splitting for RGB images
├── train.py                  # Training pipeline for RGB images
├── reproduce.py              # Reproducibility verification for RGB models
├── ms_dataset.py             # Data preparation and splitting for multispectral images
├── ms_train.py               # Training pipeline for multispectral images
├── ms_reproduce.py           # Reproducibility verification for multispectral models
├── results/                  # Directory for storing model outputs (not included in repo)
│   ├── best_model_ms.pth     # Trained model for multispectral images
│   ├── test_logits.pt        # Logits for RGB images test set
│   ├── test_logits_ms.pt     # Logits for multispectral images test set
├── EuroSAT_RGB.zip           # RGB dataset (download separately)
├── EuroSAT_MS.zip            # Multispectral dataset (download separately)
└── README.md                 # This file
```

## Setup and Requirements

### Dependencies
- Python 3.7+
- PyTorch (with CUDA support recommended)
- Torchvision
- NumPy
- Scikit-learn
- Matplotlib
- Skimage
- Tqdm

### Installation
```bash
# Clone the repository
git clone https://github.com/t-assylkhan/eurosat-classification.git
cd eurosat-classification

# Install required packages
pip install torch torchvision numpy scikit-learn matplotlib scikit-image tqdm
```

## Model Files
The trained model files are not included in this repository due to their size. To download them:

1. Go to the [Releases](https://github.com/yourusername/eurosat-classification/releases) section of this repository
2. Download the following files from the latest release:
   - `best_model_ms.pth` - The trained model for multispectral images
   - `test_logits.pt` - Logits for RGB images test set
   - `test_logits_ms.pt` - Logits for multispectral images test set

3. Place these files in the `results/` directory in your local copy of this repository:
```bash
mkdir -p results
# Move downloaded files to the results directory
```

## Dataset Access
The EuroSAT dataset can be downloaded from:
- RGB data (~91MB): https://isgwww.cs.uni-magdeburg.de/agcv_binder/compvis_wise2425/datasets_shared/EuroSAT_RGB.zip
- Multispectral data (~2.0GB): https://isgwww.cs.uni-magdeburg.de/agcv_binder/compvis_wise2425/datasets_shared/EuroSAT_MS.zip

Place the downloaded zip files in the project root directory.

## Usage Instructions

### For RGB Images
```bash
# Step 1: Split the dataset (creates train/val/test splits)
python data_splitting.py

# Step 2: Train the model (optional, if you want to retrain)
python train.py

# Step 3: Verify reproduction of results using saved model
python reproduce.py
```

### For Multispectral Images
```bash
# Step 1: Split the dataset
python ms_dataset.py

# Step 2: Train the model (optional, if you want to retrain)
python ms_train.py

# Step 3: Verify reproduction of results using saved model
python ms_reproduce.py
```

## Implementation Details

### Data Splitting
- Implements stratified train-validation-test splitting with:
  - Training set: ~2700 images
  - Validation set: ~1000 images
  - Test set: ~2000 images
- Uses a configurable seed for reproducibility (20250108024256)
- Verifies that splits are disjoint to prevent data leakage

### Model Architecture
- **RGB Classification**: Modified ResNet18 architecture optimized for 64x64 satellite images
  - Replacement of first convolution layer
  - Removal of maxpool layer
  - Fine-tuning of all layers

- **Multispectral Classification**: Dual ResNet18 architecture with feature fusion
  - Processes two groups of 3 bands each
  - Concatenates features before final classification
  - Handles variable band selection from the 13 available bands

### Training
- Multiple data augmentation strategies implemented
- Transfer learning with pretrained weights
- Learning rate scheduling
- Class weighting for imbalanced data
- Detailed metrics tracking

### Evaluation
- Class-wise accuracy metrics
- Top-5 and Bottom-5 prediction analysis
- Logit comparison for reproducibility verification

## Results
The project demonstrates successful classification of satellite imagery using both RGB and multispectral data. Performance graphs and detailed metrics are saved during training, and the best models are selected based on validation accuracy.
