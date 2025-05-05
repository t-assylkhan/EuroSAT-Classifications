import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import numpy as np

from ms_dataset import MSEuroSATDataset, get_ms_transforms, unzip_dataset

# Seed handling
ORIGINAL_SEED = 20250108024256  # 14 digits YYYYMMDDHHMMSS format
SEED = ORIGINAL_SEED % (2**32 - 1)  # Actual seed used

def set_deterministic_mode():
    """Set all operations to be deterministic."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

class OptimizedMSResNet(nn.Module):
    """Optimized ResNet18 for multispectral images with improved stability"""
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        
        # Set deterministic initialization
        torch.manual_seed(SEED)
        
        # Create two ResNet18 feature extractors
        self.resnet1 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet2 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Reset seed after loading pretrained weights
        torch.manual_seed(SEED)
        
        # Modify first conv layers for 64x64 images
        self.resnet1.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet2.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove maxpool layers
        self.resnet1.maxpool = nn.Identity()
        self.resnet2.maxpool = nn.Identity()
        
        # Get feature size
        self.feature_size = self.resnet1.fc.in_features  # 512 for ResNet18
        self.resnet1.fc = nn.Identity()
        self.resnet2.fc = nn.Identity()
        
        # Simpler feature fusion with stronger regularization
        self.fusion = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_size * 2, num_classes)
        )
        
    def forward(self, x1, x2):
        # Extract features from both paths
        f1 = self.resnet1(x1)
        f2 = self.resnet2(x2)
        
        # Concatenate features
        combined = torch.cat([f1, f2], dim=1)
        
        # Final classification
        return self.fusion(combined)

def verify_reproduction(device='cuda', tolerance=1e-4):
    """Verify that the model produces the same logits when run again."""
    print("Starting reproduction verification...")
    
    # Set deterministic mode
    set_deterministic_mode()
    
    # Unzip dataset if needed
    dataset_root = unzip_dataset('./EuroSAT_MS.zip', './datasets')
    
    # Load training info to get band selection
    try:
        with open('results/training_config_ms.json', 'r') as f:
            training_info = json.load(f)
        selected_bands = training_info['config']['bands']
    except FileNotFoundError:
        print("Warning: Training info not found. Using default bands.")
        selected_bands = [2, 3, 4, 1, 8, 11]  # RGB + Coastal, NIR, SWIR1
    
    # Load test dataset
    test_dataset = MSEuroSATDataset(
        dataset_root=dataset_root,
        split_file='./ms_data_splits/test_files.json',
        transform=get_ms_transforms(),
        bands=selected_bands
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=24,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load the saved model
    print("Loading saved model...")
    checkpoint = torch.load('results/best_model_ms.pth', map_location=device)
    model = OptimizedMSResNet(
        num_classes=checkpoint['num_classes'],
        dropout_rate=checkpoint['dropout_rate']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get new logits
    print("Computing new logits...")
    new_logits = []
    filenames = []
    with torch.no_grad():
        for img1, img2, _, batch_filenames in tqdm(test_loader):
            img1, img2 = img1.to(device), img2.to(device)
            outputs = model(img1, img2)
            new_logits.append(outputs.cpu())
            filenames.extend(batch_filenames)
    
    new_logits = torch.cat(new_logits, dim=0)
    
    # Load original logits
    print("Loading saved logits...")
    try:
        saved_data = torch.load('results/test_logits_ms.pt')
        saved_logits = saved_data['logits']
        saved_filenames = saved_data['filenames']
    except FileNotFoundError:
        print("Error: Original logits file not found!")
        return False
    
    # Verify order matches
    if filenames != saved_filenames:
        print("Warning: File order doesn't match! Reordering saved logits...")
        saved_file_to_idx = {fname: idx for idx, fname in enumerate(saved_filenames)}
        try:
            reordered_indices = [saved_file_to_idx[fname] for fname in filenames]
            saved_logits = saved_logits[reordered_indices]
        except KeyError:
            print("Error: Cannot match all filenames. Reproduction check failed.")
            return False
    
    # Compare logits and predictions
    abs_diff = torch.abs(new_logits - saved_logits)
    max_diff = float(torch.max(abs_diff))
    mean_diff = float(torch.mean(abs_diff))
    std_diff = float(torch.std(abs_diff))
    
    # Compare predictions
    new_preds = torch.argmax(new_logits, dim=1)
    saved_preds = torch.argmax(saved_logits, dim=1)
    pred_match = (new_preds == saved_preds).float().mean().item()
    
    # Compute per-class prediction matches
    class_matches = {}
    for class_name in test_dataset.classes:
        class_mask = [os.path.dirname(f) == class_name for f in filenames]
        class_new_preds = new_preds[class_mask]
        class_saved_preds = saved_preds[class_mask]
        class_match_rate = (class_new_preds == class_saved_preds).float().mean().item()
        class_matches[class_name] = class_match_rate * 100
    
    print(f"\nLogit comparison results:")
    print(f"Maximum absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Standard deviation of difference: {std_diff:.6f}")
    print(f"Overall prediction match rate: {pred_match * 100:.2f}%")
    print("\nPer-class prediction match rates:")
    for class_name, match_rate in class_matches.items():
        print(f"{class_name}: {match_rate:.2f}%")
    
    # Save detailed report
    report = {
        'reproduction_successful': max_diff <= tolerance,
        'max_difference': float(max_diff),
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),
        'prediction_match_rate': pred_match * 100,
        'class_match_rates': class_matches,
        'tolerance_used': tolerance,
        'seed_used': SEED,
        'original_seed': ORIGINAL_SEED,
        'selected_bands': selected_bands,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/reproduction_report_ms.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n✓ Saved detailed reproduction report to results/reproduction_report_ms.json")
    
    if max_diff <= tolerance:
        print("\n✓ Reproduction successful! Logits match within tolerance.")
    else:
        print("\n❌ Reproduction failed! Logits differ significantly.")
        print("Check reproduction_report_ms.json for detailed analysis.")
        
        # Print details about largest differences
        max_diff_indices = torch.topk(abs_diff.view(-1), k=5).indices
        print("\nTop 5 largest differences:")
        for idx in max_diff_indices:
            batch_idx = idx // new_logits.size(1)
            class_idx = idx % new_logits.size(1)
            filename = filenames[batch_idx]
            true_class = os.path.dirname(filename)
            new_pred_class = test_dataset.classes[new_preds[batch_idx]]
            saved_pred_class = test_dataset.classes[saved_preds[batch_idx]]
            
            print(f"\nFile: {filename}")
            print(f"True class: {true_class}")
            print(f"New logit: {new_logits[batch_idx, class_idx]:.4f}")
            print(f"Saved logit: {saved_logits[batch_idx, class_idx]:.4f}")
            print(f"Difference: {abs_diff[batch_idx, class_idx]:.4f}")
            print(f"New prediction: {new_pred_class}")
            print(f"Saved prediction: {saved_pred_class}")
    
    return max_diff <= tolerance

def main():
    """Main function to run reproduction verification"""
    print("\nStarting reproduction verification process...")
    
    # Check for required files
    required_files = [
        './results/best_model_ms.pth',
        './results/test_logits_ms.pt',
        './ms_data_splits/test_files.json',
        './EuroSAT_MS.zip'
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