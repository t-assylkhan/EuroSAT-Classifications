import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
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

def calculate_class_weights(train_dataset):
    """Calculate balanced class weights."""
    class_counts = torch.zeros(len(train_dataset.classes))
    for _, _, label, _ in train_dataset:
        class_counts[label] += 1
    
    # Calculate weights with smoothing to avoid extreme values
    weights = 1.0 / (class_counts + 1)  # Add 1 for smoothing
    weights = weights / weights.sum() * len(train_dataset.classes)
    return weights

def evaluate_test_set(model, device, config):
    """Evaluate model on test set and save logits."""
    model.eval()
    torch.set_grad_enabled(False)
    
    test_dataset = MSEuroSATDataset(
        dataset_root='./datasets/EuroSAT_MS',
        split_file='./ms_data_splits/test_files.json',
        transform=get_ms_transforms(),
        bands=config['bands']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    all_logits = []
    all_filenames = []
    
    print("\nComputing test set logits...")
    with torch.no_grad():
        for img1, img2, _, filenames in tqdm(test_loader):
            img1, img2 = img1.to(device), img2.to(device)
            outputs = model(img1, img2)
            all_logits.append(outputs.cpu())
            all_filenames.extend(filenames)
    
    all_logits = torch.cat(all_logits, dim=0)
    
    os.makedirs('results', exist_ok=True)
    torch.save({
        'logits': all_logits,
        'filenames': all_filenames
    }, 'results/test_logits_ms.pt')
    
    print("✓ Saved test logits to results/test_logits_ms.pt")

def main():
    # Set deterministic mode
    set_deterministic_mode()
    
    # Setup paths
    dataset_zip = './EuroSAT_MS.zip'
    dataset_root = unzip_dataset(dataset_zip, './datasets')
    project_root = '.'
    
    # Create directories
    os.makedirs(os.path.join(project_root, 'results'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'ms_data_splits'), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Training settings
    config = {
    'batch_size': 24,
    'learning_rate': 0.0003,
    'num_epochs': 20,
    'weight_decay': 5e-4,  # Increase weight decay
    'dropout_rate': 0.5,   # Increase dropout
    'bands': [2, 3, 4, 1, 8, 11]
    }
    
    # Setup datasets
    train_dataset = MSEuroSATDataset(
        dataset_root=dataset_root,
        split_file='./ms_data_splits/train_files.json',
        transform=get_ms_transforms(),
        bands=config['bands']
    )
    
    val_dataset = MSEuroSATDataset(
        dataset_root=dataset_root,
        split_file='./ms_data_splits/val_files.json',
        transform=get_ms_transforms(),
        bands=config['bands']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = OptimizedMSResNet(
        num_classes=len(train_dataset.classes),
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Calculate class weights for balanced training
    class_weights = calculate_class_weights(train_dataset).to(device)
    
    # Setup optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Learning rate scheduler with warm-up and cosine decay
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # First restart happens after 5 epochs
        T_mult=2,  # Each restart doubles the cycle length
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Save configuration
    training_config = {
        'seed': SEED,
        'original_seed': ORIGINAL_SEED,
        'config': config
    }
    
    with open(os.path.join('results', 'training_config_ms.json'), 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Training metrics tracking
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'class_accs': [],
        'class_names': train_dataset.classes
    }
    
    best_val_acc = 0
    best_model_state = None
    
    print("Starting model training...")
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for img1, img2, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            img1, img2 = img1.to(device), img2.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        class_correct = torch.zeros(len(train_dataset.classes))
        class_total = torch.zeros(len(train_dataset.classes))
        
        with torch.no_grad():
            for img1, img2, labels, _ in tqdm(val_loader, desc='Validating'):
                img1, img2 = img1.to(device), img2.to(device)
                labels = labels.to(device)
                
                outputs = model(img1, img2)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                for label, pred in zip(labels, predicted):
                    if label == pred:
                        class_correct[label] += 1
                    class_total[label] += 1
        
        val_acc = 100. * correct / total
        val_loss /= len(val_loader)
        
        # Calculate per-class accuracies
        class_accs = 100. * class_correct / class_total
        
        # Store metrics
        metrics['train_losses'].append(train_loss)
        metrics['val_losses'].append(val_loss)
        metrics['train_accs'].append(train_acc)
        metrics['val_accs'].append(val_acc)
        metrics['class_accs'].append(class_accs.numpy())
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Print per-class accuracies
        for i, (class_name, acc) in enumerate(zip(train_dataset.classes, class_accs)):
            print(f'{class_name}: {acc:.2f}%')
        
        # Update scheduler
        scheduler.step()
        
        # Update best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'dropout_rate': config['dropout_rate'],
                'num_classes': len(train_dataset.classes)
            }
            torch.save(best_model_state, os.path.join('results', 'best_model_ms.pth'))
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_losses'], label='Train Loss')
    plt.plot(metrics['val_losses'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_accs'], label='Train Accuracy')
    plt.plot(metrics['val_accs'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'training_curves_ms.png'))
    plt.close()
    
    # Save metrics
    for key in metrics:
        if key == 'class_accs':
            metrics[key] = [class_acc.tolist() if hasattr(class_acc, 'tolist') else class_acc 
                          for class_acc in metrics[key]]

    with open(os.path.join('results', 'training_metrics_ms.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Evaluate test set using best model
    model.load_state_dict(best_model_state['model_state_dict'])
    evaluate_test_set(model, device, config)

if __name__ == "__main__":
    main()