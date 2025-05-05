import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json
from tqdm import tqdm
import datetime

# Use same seed as training
ORIGINAL_SEED = 20250108024256  # 14 digits YYYYMMDDHHMMSS format
SEED = ORIGINAL_SEED % (2**32 - 1)  # Actual seed used

class EuroSATDataset(Dataset):
    def __init__(self, dataset_root, split_file, transform=None):
        self.dataset_root = dataset_root
        self.transform = transform
        
        with open(split_file, 'r') as f:
            self.image_files = json.load(f)
        
        self.classes = sorted(list(set([os.path.dirname(f) for f in self.image_files])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_root, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.class_to_idx[os.path.dirname(self.image_files[idx])]
        return image, label, self.image_files[idx]

def get_transform():
    """Get basic transform for testing"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_model(num_classes):
    """Get ResNet18 model modified for 64x64 images"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify first convolution layer for 64x64 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove maxpool layer which would reduce spatial dimensions too much for 64x64
    model.maxpool = nn.Identity()
    
    # Modify final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def verify_reproduction(device='cuda', tolerance=1e-5):
    """Verify that the model produces the same logits when run again."""
    print("Starting reproduction verification...")
    
    # Set seeds
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # Load test dataset
    test_dataset = EuroSATDataset(
        dataset_root='./EuroSAT_RGB',
        split_file='./data_splits/test_files.json',
        transform=get_transform()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,  # Important: keep order consistent
        num_workers=4,
        pin_memory=True
    )
    
    # Load the saved model
    print("Loading saved model...")
    checkpoint = torch.load('results/best_model.pth', map_location=device)
    model = get_model(num_classes=len(test_dataset.classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get new logits
    print("Computing new logits...")
    new_logits = []
    filenames = []
    with torch.no_grad():
        for inputs, _, batch_filenames in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            new_logits.append(outputs.cpu())
            filenames.extend(batch_filenames)
    
    new_logits = torch.cat(new_logits, dim=0)
    
    # Load original logits
    print("Loading saved logits...")
    saved_data = torch.load('results/test_logits.pt')
    saved_logits = saved_data['logits']
    saved_filenames = saved_data['filenames']
    
    # Verify order matches
    if filenames != saved_filenames:
        print("Warning: File order doesn't match! Reordering saved logits...")
        # Create mapping from filename to index
        saved_file_to_idx = {fname: idx for idx, fname in enumerate(saved_filenames)}
        # Reorder saved logits to match current order
        reordered_indices = [saved_file_to_idx[fname] for fname in filenames]
        saved_logits = saved_logits[reordered_indices]
    
    # Compare logits
    abs_diff = torch.abs(new_logits - saved_logits)
    max_diff = float(torch.max(abs_diff))
    mean_diff = float(torch.mean(abs_diff))
    std_diff = float(torch.std(abs_diff))
    
    print(f"\nLogit comparison results:")
    print(f"Maximum absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Standard deviation of difference: {std_diff:.6f}")
    
    # Compute per-class differences
    class_diffs = {}
    for class_name in test_dataset.classes:
        class_idx = test_dataset.class_to_idx[class_name]
        class_mask = [os.path.dirname(f) == class_name for f in filenames]
        class_abs_diff = abs_diff[class_mask]
        
        class_diffs[class_name] = {
            'max_diff': float(torch.max(class_abs_diff)),
            'mean_diff': float(torch.mean(class_abs_diff)),
            'std_diff': float(torch.std(class_abs_diff))
        }
    
    # Check if differences are within acceptable threshold
    reproduction_passed = max_diff <= tolerance
    
    # Save detailed report
    report = {
        'reproduction_successful': reproduction_passed,
        'max_difference': float(max_diff),
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),
        'per_class_differences': class_diffs,
        'tolerance_used': tolerance,
        'seed_used': SEED,
        'original_seed': ORIGINAL_SEED,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/reproduction_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n✓ Saved detailed reproduction report to results/reproduction_report.json")
    
    if reproduction_passed:
        print("\n✓ Reproduction successful! Logits match within tolerance.")
    else:
        print("\n❌ Reproduction failed! Logits differ significantly.")
        print("Check reproduction_report.json for detailed analysis.")
    
    return reproduction_passed

def main():
    """Main function to run reproduction verification"""
    print("\nStarting reproduction verification process...")
    
    # Check for required files
    required_files = [
        './results/best_model.pth',
        './results/test_logits.pt',
        './data_splits/test_files.json'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(
                f"Required file not found: {file}\n"
                "Please ensure training has completed successfully."
            )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("Warning: Running on CPU. This is fine for reproduction.")
    
    # Run verification
    verify_reproduction(device)

if __name__ == "__main__":
    main()