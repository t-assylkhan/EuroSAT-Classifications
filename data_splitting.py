import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import zipfile
import shutil

def unpack_dataset(zip_path, extract_path):
    """
    Unpack the EuroSAT RGB dataset zip file.
    
    Args:
        zip_path (str): Path to the EuroSAT_RGB.zip file
        extract_path (str): Path where to extract the files
    """
    print(f"\nUnpacking dataset from: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✓ Dataset unpacked successfully!")
    except FileNotFoundError:
        print(f"Error: ZIP file not found at {zip_path}")
        raise
    except Exception as e:
        print(f"Error unpacking dataset: {str(e)}")
        raise

def setup_directories(project_root):
    """Create necessary directories."""
    dirs = {
        'data_splits': os.path.join(project_root, 'data_splits'),
        'results': os.path.join(project_root, 'results'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    return dirs

def create_splits(dataset_root, output_dir, seed):
    """
    Create train-val-test splits of the dataset.
    Using minimum required sizes:
    - Train: 2700 images
    - Val: 1000 images
    - Test: 2000 images
    
    Args:
        dataset_root (str): Path to the EuroSAT_RGB directory
        output_dir (str): Directory to save the split files
        seed (int): Manual seed for reproducibility
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
        'train': 2700 // n_classes,  # ~450 per class
        'val': 1000 // n_classes,    # ~167 per class
        'test': 2000 // n_classes    # ~333 per class
    }
    
    print("\nProcessing classes:")
    # Process each class separately (stratified split)
    for class_name in classes:
        print(f"\nClass: {class_name}")
        class_dir = os.path.join(dataset_root, class_name)
        
        # Get all images for this class
        files = [os.path.join(class_name, f) for f in os.listdir(class_dir) 
                if f.endswith('.jpg')]
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
    
    # Verify splits are disjoint and have correct sizes
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
    
    print("✓ All splits are disjoint")
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

def main():
    # Set paths
    project_root = "."  # Current directory
    dataset_zip = "./EuroSAT_RGB.zip"
    dataset_dir = "./EuroSAT_RGB"
    
    # Set manual seed (14-20 digits as required)
    original_seed = 20250108024256  # YYYYMMDDHHMMSS format
    seed = original_seed % (2**32 - 1)  # Ensure within numpy/torch limits
    
    print("Starting data preparation...")
    
    # Create directories
    dirs = setup_directories(project_root)
    
    # Unpack dataset if needed
    if not os.path.exists(dataset_dir) and os.path.exists(dataset_zip):
        unpack_dataset(dataset_zip, project_root)
    
    # Create and save splits
    splits = create_splits(
        dataset_root=dataset_dir,
        output_dir=dirs['data_splits'],
        seed=seed
    )
    
    print("\nData splitting completed successfully!")

if __name__ == "__main__":
    main()