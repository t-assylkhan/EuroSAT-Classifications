import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.io import imread
import numpy as np
import zipfile
import shutil

# Seed handling
ORIGINAL_SEED = 20250108024256  # 14 digits YYYYMMDDHHMMSS format
SEED = ORIGINAL_SEED % (2**32 - 1)  # Actual seed used

def unzip_dataset(zip_path, extract_path):
    """Unzip the dataset if it hasn't been extracted already."""
    os.makedirs(extract_path, exist_ok=True)
    
    # Check if dataset is already extracted
    dataset_marker = os.path.join(extract_path, 'EuroSAT_MS')
    if os.path.exists(dataset_marker):
        print(f"Dataset already extracted to {extract_path}")
        return dataset_marker
    
    # Unzip the file
    print(f"Extracting dataset from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    return os.path.join(extract_path, 'EuroSAT_MS')

def get_ms_transforms():
    """Get transforms for multispectral images."""
    return transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

def setup_directories(project_root):
    """Create necessary directories."""
    dirs = {
        'data_splits': os.path.join(project_root, 'ms_data_splits'),
        'results': os.path.join(project_root, 'results'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    return dirs

def create_splits(dataset_root, output_dir, seed=SEED):
    """
    Create train-val-test splits of the dataset.
    Using optimized sizes that still meet requirements:
    - Train: ~3000 images (~500 per class)
    - Val: ~1200 images (~200 per class)
    - Test: ~2400 images (~400 per class)
    """
    print(f"\nCreating splits with seed: {seed}")
    np.random.seed(seed)
    
    # Initialize splits
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    # Get class directories
    classes = [d for d in os.listdir(dataset_root) 
              if os.path.isdir(os.path.join(dataset_root, d))]
    classes.sort()  # Ensure consistent ordering
    
    # Calculate per-class sample sizes (stratified)
    n_classes = len(classes)
    samples_per_class = {
        'train': 3000 // n_classes,  # ~500 per class
        'val': 1200 // n_classes,    # ~200 per class
        'test': 2400 // n_classes    # ~400 per class
    }
    
    print("\nProcessing classes:")
    # Process each class separately (stratified split)
    for class_name in classes:
        print(f"\nClass: {class_name}")
        class_dir = os.path.join(dataset_root, class_name)
        
        # Get all images for this class
        files = [os.path.join(class_name, f) for f in os.listdir(class_dir) 
                if f.endswith('.tif')]
        np.random.shuffle(files)  # Shuffle files
        
        print(f"Total images available: {len(files)}")
        
        # Take required number of samples for each split
        current_idx = 0
        
        # Training samples
        train_end = current_idx + samples_per_class['train']
        splits['train'].extend(files[current_idx:train_end])
        current_idx = train_end
        
        # Validation samples
        val_end = current_idx + samples_per_class['val']
        splits['val'].extend(files[current_idx:val_end])
        current_idx = val_end
        
        # Test samples
        test_end = current_idx + samples_per_class['test']
        splits['test'].extend(files[current_idx:test_end])
        
        print(f"Train: {samples_per_class['train']} images")
        print(f"Val: {samples_per_class['val']} images")
        print(f"Test: {samples_per_class['test']} images")
    
    # Verify splits are disjoint and meet requirements
    verify_splits(splits)
    
    # Save splits
    save_splits(splits, output_dir, seed)
    
    return splits

def verify_splits(splits):
    """Verify that splits are disjoint and meet size requirements."""
    print("\nVerifying splits...")
    
    # Convert to sets for intersection checks
    train_set = set(splits['train'])
    val_set = set(splits['val'])
    test_set = set(splits['test'])
    
    # Check for overlaps
    train_val = train_set.intersection(val_set)
    train_test = train_set.intersection(test_set)
    val_test = val_set.intersection(test_set)
    
    if len(train_val) > 0 or len(train_test) > 0 or len(val_test) > 0:
        raise ValueError("Splits are not disjoint!")
    
    # Verify minimum sizes
    min_sizes = {
        'train': 2700,
        'val': 1000,
        'test': 2000
    }
    
    for split_name, min_size in min_sizes.items():
        actual_size = len(splits[split_name])
        if actual_size < min_size:
            raise ValueError(
                f"{split_name} split has {actual_size} samples, "
                f"but minimum required is {min_size}"
            )
    
    print("✓ All splits are disjoint and meet minimum size requirements")
    print(f"\nFinal split sizes:")
    print(f"Train: {len(splits['train'])} images")
    print(f"Val: {len(splits['val'])} images")
    print(f"Test: {len(splits['test'])} images")

def save_splits(splits, output_dir, seed):
    """Save splits and metadata to files."""
    print("\nSaving splits...")
    
    # Save each split to a separate file
    for split_name, files in splits.items():
        output_file = os.path.join(output_dir, f'{split_name}_files.json')
        with open(output_file, 'w') as f:
            json.dump(files, f, indent=2)
        print(f"✓ Saved {split_name} split to: {output_file}")
    
    # Save split info
    info = {
        'seed': seed,
        'train_size': len(splits['train']),
        'val_size': len(splits['val']),
        'test_size': len(splits['test']),
        'min_required': {
            'train': 2700,
            'val': 1000,
            'test': 2000
        }
    }
    
    info_file = os.path.join(output_dir, 'split_info.json')
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"✓ Saved split info to: {info_file}")

class MSEuroSATDataset(Dataset):
    """Dataset class for EuroSAT multispectral data."""
    def __init__(self, dataset_root, split_file, transform=None, bands=[2,3,4,1,8,11]):
        """
        Args:
            dataset_root: Root directory of the EuroSAT_MS dataset
            split_file: Path to the JSON file containing train/val/test splits
            transform: Optional transform to be applied on the image
            bands: List of bands to use. Default uses:
                  [2,3,4] = RGB
                  [1,8,11] = Coastal, NIR, SWIR1
        """
        self.dataset_root = dataset_root
        self.transform = transform
        self.bands = bands
        
        with open(split_file, 'r') as f:
            self.image_files = json.load(f)
        
        self.classes = sorted(list(set([os.path.dirname(f) for f in self.image_files])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Verify bands selection
        if len(bands) != 6:
            raise ValueError("Must select exactly 6 bands")
        if not all(0 <= b <= 12 for b in bands):
            raise ValueError("Band indices must be between 0 and 12")
        if len(set(bands)) != 6:
            raise ValueError("Band indices must be unique")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_root, self.image_files[idx])
        
        # Read TIFF using skimage
        img = imread(img_path)
        
        # Select and split into two groups of 3 bands
        img1 = img[self.bands[:3]].astype(np.float32)  # First 3 bands
        img2 = img[self.bands[3:]].astype(np.float32)  # Second 3 bands
        
        # Normalize to [0, 1]
        img1 = img1 / 65535.0
        img2 = img2 / 65535.0
        
        # Convert to tensors 
        img1 = torch.from_numpy(img1.copy())
        img2 = torch.from_numpy(img2.copy())
        
        # Apply transforms if any
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        label = self.class_to_idx[os.path.dirname(self.image_files[idx])]
        return img1, img2, label, self.image_files[idx]

def prepare_dataset(zip_path='./EuroSAT_MS.zip', project_root='.'):
    """Main function to prepare the dataset."""
    print("Starting data preparation...")
    
    # Create directories
    dirs = setup_directories(project_root)
    
    # Unpack dataset
    dataset_dir = unzip_dataset(zip_path, './datasets')
    
    # Create and save splits
    splits = create_splits(
        dataset_root=dataset_dir,
        output_dir=dirs['data_splits'],
        seed=SEED
    )
    
    print("\nData preparation completed successfully!")
    return splits

if __name__ == "__main__":
    prepare_dataset()