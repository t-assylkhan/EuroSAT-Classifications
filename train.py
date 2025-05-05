import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import shutil

# Seed handling
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
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_root, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.class_to_idx[os.path.dirname(self.image_files[idx])]
        return image, label, self.image_files[idx]

def get_transforms(augmentation='default'):
    """Get transforms with different augmentation settings."""
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # Basic transform for validation/test
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Augmentation setting 1 - Mild
    augment_1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Augmentation setting 2 - Moderate
    augment_2 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])
    
    transforms_dict = {
        'default': basic_transform,
        'augment_1': augment_1,
        'augment_2': augment_2
    }
    
    return transforms_dict[augmentation]

def get_model(num_classes):
    """Get ResNet18 model modified for 64x64 images."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify first convolution layer for 64x64 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove maxpool layer which would reduce spatial dimensions too much for 64x64
    model.maxpool = nn.Identity()
    
    # Modify final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels, _ in tqdm(train_loader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = torch.zeros(len(val_loader.dataset.classes))
    class_total = torch.zeros(len(val_loader.dataset.classes))
    
    with torch.no_grad():
        for inputs, labels, _ in tqdm(val_loader, desc='Validating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for label, pred in zip(labels, predicted):
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    # Calculate per-class accuracies
    class_acc = 100. * class_correct / class_total
    
    return running_loss / len(val_loader), 100. * correct / total, class_acc

def analyze_top_bottom_images(model, test_loader, device, num_classes=3, k=5):
    """Analyze top and bottom scoring images for selected classes."""
    model.eval()
    idx_to_class = test_loader.dataset.idx_to_class
    os.makedirs('results/top_bottom_analysis', exist_ok=True)
    
    # Randomly select classes to analyze
    classes_to_analyze = torch.randperm(len(idx_to_class))[:num_classes].tolist()
    
    results = {}
    with torch.no_grad():
        for class_idx in classes_to_analyze:
            class_name = idx_to_class[class_idx]
            print(f"\nAnalyzing class: {class_name}")
            
            all_scores = []
            for inputs, labels, filenames in tqdm(test_loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                scores = outputs[:, class_idx].cpu().softmax(dim=0)
                
                for score, filename in zip(scores, filenames):
                    all_scores.append((score.item(), filename))
            
            # Sort by score
            all_scores.sort(key=lambda x: x[0])
            
            # Get top and bottom k
            bottom_k = all_scores[:k]
            top_k = all_scores[-k:]
            
            # Save images
            class_dir = os.path.join('results/top_bottom_analysis', class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Save bottom k
            for i, (score, filename) in enumerate(bottom_k):
                src = os.path.join('./EuroSAT_RGB', filename)
                dst = os.path.join(class_dir, f'bottom_{i+1}_{score:.3f}.jpg')
                shutil.copy2(src, dst)
            
            # Save top k
            for i, (score, filename) in enumerate(top_k):
                src = os.path.join('./EuroSAT_RGB', filename)
                dst = os.path.join(class_dir, f'top_{i+1}_{score:.3f}.jpg')
                shutil.copy2(src, dst)
            
            results[class_name] = {
                'top_k': [(s, f) for s, f in top_k],
                'bottom_k': [(s, f) for s, f in bottom_k]
            }
    
    return results

def plot_training_curves(metrics, save_dir, aug_setting):
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(metrics['train_losses'], label='Train')
    plt.plot(metrics['val_losses'], label='Validation')
    plt.title(f'Loss Curves ({aug_setting})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(metrics['train_accs'], label='Train')
    plt.plot(metrics['val_accs'], label='Validation')
    plt.title(f'Accuracy Curves ({aug_setting})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot per-class validation accuracy
    plt.subplot(1, 3, 3)
    class_accs = np.array(metrics['class_accs'])
    for i, class_name in enumerate(metrics['class_names']):
        plt.plot(class_accs[:, i], label=class_name)
    plt.title('Per-class Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'metrics_{aug_setting}.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, save_dir, aug_setting):
    """Train model with fine-tuning of all layers."""
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'class_accs': [],
        'class_names': val_loader.dataset.classes
    }
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        metrics['train_losses'].append(train_loss)
        metrics['train_accs'].append(train_acc)
        
        # Validate
        val_loss, val_acc, class_accs = validate(
            model, val_loader, criterion, device
        )
        metrics['val_losses'].append(val_loss)
        metrics['val_accs'].append(val_acc)
        metrics['class_accs'].append(class_accs.numpy())
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Print per-class accuracies
        for i, (class_name, acc) in enumerate(zip(val_loader.dataset.classes, class_accs)):
            print(f'{class_name}: {acc:.2f}%')
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f'Learning rate changed from {old_lr:.6f} to {new_lr:.6f}')
        
        # Save best model state
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'aug_setting': aug_setting,
                'seed': SEED
            }
            torch.save(best_model_state, os.path.join(save_dir, 'best_model.pth'))
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    # Plot training curves
    plot_training_curves(metrics, save_dir, aug_setting)
    
    return best_val_acc, best_model_state

def evaluate_test_set(model, device):
    """Evaluate model on test set and save logits."""
    model.eval()
    
    test_dataset = EuroSATDataset(
        dataset_root='./EuroSAT_RGB',
        split_file='./data_splits/test_files.json',
        transform=get_transforms('default')
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
        for inputs, labels, filenames in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_logits.append(outputs.cpu())
            all_filenames.extend(filenames)
    
    all_logits = torch.cat(all_logits, dim=0)
    
    torch.save({
        'logits': all_logits,
        'filenames': all_filenames
    }, 'results/test_logits.pt')
    
    print("✓ Saved test logits to results/test_logits.pt")
    
    print("\nAnalyzing top/bottom scoring images...")
    analysis_results = analyze_top_bottom_images(model, test_loader, device)
    
    with open('results/top_bottom_analysis/analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)

def main():
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Training settings
    config = {
        'batch_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 15,
        'weight_decay': 1e-4
    }
    
    # Keep track of best model across augmentations
    best_overall_acc = 0
    best_model_state = None
    
    # Try both augmentation settings
    for aug_setting in ['augment_1', 'augment_2']:
        print(f"\nTraining with {aug_setting}")
        
        # Create datasets
        # Create datasets
        train_dataset = EuroSATDataset(
            dataset_root='./EuroSAT_RGB',
            split_file='./data_splits/train_files.json',
            transform=get_transforms(aug_setting)
        )
        
        val_dataset = EuroSATDataset(
            dataset_root='./EuroSAT_RGB',
            split_file='./data_splits/val_files.json',
            transform=get_transforms('default')
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
        model = get_model(num_classes=len(train_dataset.classes)).to(device)
        
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Loss and scheduler
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=3,
        )
        
        # Train model
        print(f"\nStarting training with {aug_setting}...")
        val_acc, model_state = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=config['num_epochs'],
            save_dir='results',
            aug_setting=aug_setting
        )
        
        print(f"Finished training with {aug_setting}")
        print(f"Best validation accuracy: {val_acc:.2f}%")
        
        # Update best model if this one is better
        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            best_model_state = model_state
            
            # Save current best model
            torch.save(best_model_state, 'results/best_model.pth')
            print(f"New best model saved!")
    
    print(f"\nTraining completed!")
    print(f"Best overall validation accuracy: {best_overall_acc:.2f}%")
    print(f"Best augmentation setting: {best_model_state['aug_setting']}")
    
    # Load best model for final evaluation
    model = get_model(num_classes=len(train_dataset.classes)).to(device)
    model.load_state_dict(best_model_state['model_state_dict'])
    
    # Evaluate on test set and analyze top/bottom images
    print("\nEvaluating best model on test set...")
    evaluate_test_set(model, device)
    
    # Save configuration and results
    results = {
        'best_validation_accuracy': best_overall_acc,
        'best_augmentation': best_model_state['aug_setting'],
        'best_epoch': best_model_state['epoch'],
        'seed': SEED,
        'original_seed': ORIGINAL_SEED,
        'config': config
    }
    
    with open('results/training_info.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()