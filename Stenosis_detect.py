import os
import time
import json
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image, ImageDraw
from datetime import datetime
from skimage import morphology
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set up logging and constants
CURRENT_USER = os.getenv('USERNAME', 'user')
CURRENT_DATETIME = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Configure logger
logger = logging.getLogger(CURRENT_USER)
logger.setLevel(logging.INFO)

# Clear existing handlers to avoid duplicate logs
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Create file handler
log_filename = f"improved_stenosis_detection_{CURRENT_USER}_{CURRENT_DATETIME.replace(' ', '_').replace(':', '-')}.log"
fh = logging.FileHandler(log_filename)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds()

class StenosisDataset(Dataset):
    """Dataset for stenosis detection with improved handling of imbalance"""
    def __init__(self, images_dir, annotations_path, transform=None, debug=False):
        self.images_dir = images_dir
        self.transform = transform
        self.debug = debug
        
        # Load and parse annotations
        try:
            with open(annotations_path) as f:
                self.annotations = json.load(f)
                
            logger.info(f"Successfully loaded annotations from {annotations_path}")
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            raise
            
        # Extract image IDs and filenames
        self.image_ids = [img['id'] for img in self.annotations['images']]
        self.image_filenames = {img['id']: img['file_name'] for img in self.annotations['images']}
        
        # Create category ID to name mapping
        self.categories = {cat['id']: cat['name'] for cat in self.annotations['categories']}
        
        # Map stenosis category
        self.stenosis_category_id = None
        for cat_id, cat_name in self.categories.items():
            if cat_name.lower() == 'stenosis':
                self.stenosis_category_id = cat_id
                break
        
        if self.stenosis_category_id is None:
            logger.warning("No explicit 'stenosis' category found. Using all annotations as stenosis.")
        
        logger.info(f"Stenosis category ID: {self.stenosis_category_id}")
        
        # Create image ID to annotations mapping
        self.image_annotations = {}
        self.stenosis_labels = {}
        
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
            
            # Mark as stenosis if any annotation has the stenosis category
            # or if we're treating all annotations as stenosis
            if self.stenosis_category_id is None or ann.get('category_id') == self.stenosis_category_id:
                self.stenosis_labels[img_id] = 1  # Has stenosis
        
        # Calculate class distribution
        stenosis_count = len(self.stenosis_labels)
        normal_count = len(self.image_ids) - stenosis_count
        
        logger.info(f"Loaded {len(self.image_ids)} images")
        logger.info(f"Class distribution: Normal: {normal_count}, Stenosis: {stenosis_count}")
        
        # Separate stenosis and normal images for balanced sampling if needed
        self.stenosis_indices = [i for i, img_id in enumerate(self.image_ids) if img_id in self.stenosis_labels]
        self.normal_indices = [i for i, img_id in enumerate(self.image_ids) if img_id not in self.stenosis_labels]
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, self.image_filenames[img_id])
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black dummy image in case of error
            image = Image.new('RGB', (256, 256), color='black')
        
        # Create binary mask from annotations (for visualization or potential multi-task learning)
        mask = self.create_mask(img_id, image.size)
        
        # Determine stenosis label (binary: 0=normal, 1=stenosis)
        stenosis_label = 1 if img_id in self.stenosis_labels else 0
        
        # Apply transformations
        if self.transform:
            # For albumentations
            if isinstance(self.transform, A.Compose):
                transformed = self.transform(image=np.array(image), mask=np.array(mask))
                image = transformed['image']
                mask = transformed['mask']
            # For torchvision transforms
            else:
                image = self.transform(image)
                mask = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)
        else:
            # Convert to tensors if no transform
            image = torch.from_numpy(np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0)
            mask = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)
        
        return image, mask, stenosis_label, img_id
    
    def create_mask(self, img_id, img_size):
        """Create a binary mask from annotations showing stenosis regions"""
        width, height = img_size
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # If image has no annotations, return empty mask
        if img_id not in self.image_annotations:
            return mask
        
        # Draw each annotation on the mask
        for ann in self.image_annotations[img_id]:
            # Only include annotations with the stenosis category if we have a specific ID
            if self.stenosis_category_id is not None and ann.get('category_id') != self.stenosis_category_id:
                continue
                
            if 'segmentation' in ann and len(ann['segmentation']) > 0:
                for segment in ann['segmentation']:
                    # Convert segment points to polygon array
                    polygon = np.array(segment).reshape((-1, 2))
                    
                    # Skip invalid polygons
                    if len(polygon) < 3:
                        continue
                    
                    # Draw polygon on mask
                    mask = self._draw_polygon(mask, polygon)
        
        return mask
    
    def _draw_polygon(self, mask, polygon):
        """Draw a polygon on the mask using PIL for reliability"""
        # Create a temporary mask
        temp_mask = Image.new('L', (mask.shape[1], mask.shape[0]), 0)
        draw = ImageDraw.Draw(temp_mask)
        
        # Convert polygon points to tuple list for PIL
        polygon_list = [(x, y) for x, y in polygon]
        
        # Draw and fill the polygon
        if len(polygon_list) > 2:
            draw.polygon(polygon_list, outline=1, fill=1)
        
        # Convert back to numpy array and add to main mask
        poly_mask = np.array(temp_mask)
        mask = np.maximum(mask, poly_mask)
        
        return mask
    
    def debug_dataset(self, num_samples=5, save_dir=None):
        """Print debug information about the dataset and save sample visualizations"""
        # Create save directory if provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Get a mix of stenosis and normal samples if available
        normal_samples = min(num_samples // 2, len(self.normal_indices))
        stenosis_samples = min(num_samples - normal_samples, len(self.stenosis_indices))
        
        # If one class is missing, use all samples from the other
        if normal_samples == 0:
            stenosis_samples = min(num_samples, len(self.stenosis_indices))
        elif stenosis_samples == 0:
            normal_samples = min(num_samples, len(self.normal_indices))
        
        # Get indices for visualization
        indices = []
        if normal_samples > 0:
            indices.extend(random.sample(self.normal_indices, normal_samples))
        if stenosis_samples > 0:
            indices.extend(random.sample(self.stenosis_indices, stenosis_samples))
        
        # Shuffle the indices
        random.shuffle(indices)
        
        # Setup visualization
        fig, axes = plt.subplots(len(indices), 2, figsize=(10, 5 * len(indices)))
        if len(indices) == 1:
            axes = np.array([axes])
        
        # Process each sample
        for i, idx in enumerate(indices):
            img_id = self.image_ids[idx]
            img_path = os.path.join(self.images_dir, self.image_filenames[img_id])
            
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Create mask
                mask = self.create_mask(img_id, img.size)
                
                # Determine label
                stenosis_label = 1 if img_id in self.stenosis_labels else 0
                
                # Calculate mask statistics
                mask_pixels = np.sum(mask > 0)
                total_pixels = mask.shape[0] * mask.shape[1]
                mask_percentage = 100 * mask_pixels / total_pixels if total_pixels > 0 else 0
                
                # Log information
                logger.info(f"Sample {i+1} - ID: {img_id}")
                logger.info(f"  Image Shape: {img.size}")
                logger.info(f"  Stenosis Label: {'Yes' if stenosis_label else 'No'}")
                logger.info(f"  Mask Coverage: {mask_pixels} pixels ({mask_percentage:.3f}%)")
                if img_id in self.image_annotations:
                    logger.info(f"  Annotations: {len(self.image_annotations[img_id])}")
                
                # Plot image
                axes[i, 0].imshow(img)
                axes[i, 0].set_title(f"Image (ID: {img_id}, Label: {'Stenosis' if stenosis_label else 'Normal'})")
                axes[i, 0].axis('off')
                
                # Plot mask
                axes[i, 1].imshow(mask, cmap='gray')
                axes[i, 1].set_title(f"Mask (Coverage: {mask_percentage:.2f}%)")
                axes[i, 1].axis('off')
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                
                # Show placeholder for error
                axes[i, 0].text(0.5, 0.5, f"Error loading image: {e}", 
                               ha='center', va='center', wrap=True)
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f"dataset_samples_{CURRENT_DATETIME.replace(' ', '_').replace(':', '-')}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved dataset visualization to {save_path}")
        
        plt.close()


class ModifiedResNet(nn.Module):
    """Modified ResNet for stenosis detection with attention mechanism"""
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.5):
        super(ModifiedResNet, self).__init__()
        
        # Load pretrained ResNet-50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Replace last layer
        self.resnet.fc = nn.Identity()
        
        # Features dimension after ResNet-50 (2048)
        feat_dim = 2048
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classifier with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Extract features from ResNet backbone
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Global average pooling
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        return x


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)  # pt = probability of being target class
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)


def get_transforms(img_size=224):
    """Get transforms using Albumentations for better augmentation"""
    # Training transforms with more aggressive augmentation
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.CoarseDropout(max_holes=10, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Validation transforms (no augmentation)
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # TTA transform for test-time augmentation
    tta_transforms = [
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=90, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    ]
    
    return train_transform, val_transform, tta_transforms


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch with gradient accumulation for large batch training"""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    for i, (images, _, labels, _) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Convert to one-hot encoding for focal loss
        if isinstance(criterion, FocalLoss):
            one_hot_labels = torch.zeros(labels.size(0), 2, device=device)
            one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        if isinstance(criterion, FocalLoss):
            loss = criterion(outputs, one_hot_labels)
        else:
            loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        
        # Step optimizer
        optimizer.step()
        optimizer.zero_grad()
        
        # Step scheduler if OneCycleLR
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # Update metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of stenosis (class 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = running_loss / len(dataloader)
    
    return metrics


def validate(model, dataloader, criterion, device, use_tta=False, tta_transforms=None):
    """Validate the model with optional test-time augmentation"""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    all_img_ids = []
    
    with torch.no_grad():
        for images, _, labels, img_ids in dataloader:
            batch_size = images.size(0)
            
            # Standard forward pass (no TTA)
            images = images.to(device)
            outputs = model(images)
            
            labels = labels.to(device)
            
            # Convert to one-hot encoding for focal loss
            if isinstance(criterion, FocalLoss):
                one_hot_labels = torch.zeros(labels.size(0), 2, device=device)
                one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
                loss = criterion(outputs, one_hot_labels)
            else:
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_img_ids.extend(img_ids.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = running_loss / len(dataloader)
    
    return metrics, all_labels, all_preds, all_probs, all_img_ids


def calculate_metrics(labels, predictions, probabilities=None):
    """Calculate classification metrics"""
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
    }
    
    if probabilities is not None:
        try:
            # Calculate AUC if we have probabilities
            metrics['auc'] = roc_auc_score(labels, probabilities)
        except Exception as e:
            # Don't print warning for single-class case
            if "Only one class" not in str(e):
                logger.warning(f"Could not calculate AUC: {e}")
            metrics['auc'] = 0.0
    
    # Calculate confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(labels, predictions, labels=[0, 1])
    
    return metrics


def visualize_predictions(model, dataloader, device, num_samples=8, save_path=None):
    """Visualize model predictions with attention visualization (FIXED)"""
    model.eval()
    
    # Get samples
    all_images = []
    all_labels = []
    all_img_ids = []
    
    with torch.no_grad():
        for batch_idx, (images, _, labels, img_ids) in enumerate(dataloader):
            all_images.append(images)
            all_labels.append(labels)
            all_img_ids.append(img_ids)
            if len(all_images) * images.shape[0] >= num_samples:
                break
    
    if not all_images:
        logger.warning("No samples available for visualization")
        return
    
    # Concatenate batches
    images = torch.cat(all_images, dim=0)[:num_samples]
    labels = torch.cat(all_labels, dim=0)[:num_samples]
    img_ids = torch.cat(all_img_ids, dim=0)[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    # Convert to numpy
    preds = preds.cpu().numpy()
    probs_stenosis = probs[:, 1].cpu().numpy()  # Probability of stenosis (class 1)
    
    # Create visualization grid
    num_disp = min(num_samples, len(images))
    rows = min(4, num_disp)
    cols = min(4, (num_disp + rows - 1) // rows)
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(4 * cols, 3 * rows))
    
    # Handle case with a single sample
    if rows == 1 and cols == 1:
        axes = np.array([[axes[0], axes[1]]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_disp, rows * cols)):
        row = i // cols
        col = (i % cols) * 2
        
        # Get image and convert from tensor to numpy
        img = images[i].permute(1, 2, 0).cpu().numpy()
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std + mean).clip(0, 1)
        
        # Get true label and prediction
        label = labels[i].item()
        pred = preds[i]
        prob = probs_stenosis[i]
        
        # Display image
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"Image ID: {img_ids[i].item()}\nTrue: {'Stenosis' if label==1 else 'Normal'}")
        axes[row, col].axis('off')
        
        # Create prediction visualization
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('stenosis', ['blue', 'red'])
        
        # Create a probability heat map (2D array)
        prob_viz = np.ones((img.shape[0], img.shape[1])) * prob
        
        # Display prediction
        pred_label = "Stenosis" if pred == 1 else "Normal"
        correct = label == pred
        color = "green" if correct else "red"
        
        axes[row, col+1].imshow(prob_viz, cmap=cmap, vmin=0, vmax=1)
        axes[row, col+1].set_title(f"Prediction: {pred_label}\nProbability: {prob:.3f}", 
                                   color=color)
        axes[row, col+1].axis('off')
    
    # Remove empty subplots
    for i in range(min(num_samples, rows * cols), rows * cols):
        row = i // cols
        col = (i % cols) * 2
        for j in range(2):
            fig.delaxes(axes[row, col + j])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved prediction visualization to {save_path}")
    
    plt.close()


def plot_training_curves(train_metrics, val_metrics, save_path):
    """Plot training metrics curves"""
    plt.figure(figsize=(15, 10))
    
    metrics_to_plot = [
        ('loss', 'Loss'),
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1 Score'),
    ]
    
    for i, (metric_name, metric_label) in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i+1)
        
        if metric_name in train_metrics and train_metrics[metric_name]:
            plt.plot(train_metrics[metric_name], label='Train')
        if metric_name in val_metrics and val_metrics[metric_name]:
            plt.plot(val_metrics[metric_name], label='Val')
        
        plt.title(f'{metric_label} vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric_label)
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved training curves to {save_path}")
    plt.close()

# Helper function for creating compatible scheduler
def create_compatible_scheduler(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6):
    """Create a learning rate scheduler that works with different PyTorch versions"""
    try:
        # Try with verbose parameter first
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr, verbose=True
        )
        logger.info("Created ReduceLROnPlateau scheduler with verbose mode")
    except TypeError:
        # Fall back to version without verbose parameter
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr
        )
        logger.info("Created ReduceLROnPlateau scheduler without verbose mode (older PyTorch version)")
    
    return scheduler


def main():
    """Main training and evaluation function"""
    # Configuration
    config = {
        'img_size': 224,  # ResNet default input size
        'batch_size': 16,
        'num_epochs': 15,  # Reduced number of epochs since we're getting good results quickly
        'patience': 5,     # Reduced patience
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'train_split': 0.8,
        'use_class_weights': True,
        'use_focal_loss': True,
        'use_tta': False,  # Disabled TTA which was causing issues
        'balanced_sampling': True,
        'num_workers': 0,  # Set to 0 to avoid process issues
    }
    
    # File paths
    paths = {
        'images_dir': r"D:\MyFYP\Dataset\arcadfae\EXP1\arcade\arcade\stenosis\train\images",
        'annotations_path': r"D:\MyFYP\Dataset\arcadfae\EXP1\arcade\arcade\stenosis\train\annotations\train.json",
        'val_images_dir': r"D:\MyFYP\Dataset\arcadfae\EXP1\arcade\arcade\stenosis\val\images",
        'val_annotations_path': r"D:\MyFYP\Dataset\arcadfae\EXP1\arcade\arcade\stenosis\val\annotations\val.json",
    }
    
    # Create output directories
    timestamp = CURRENT_DATETIME.replace(' ', '_').replace(':', '-')
    output_dir = f"improved_stenosis_results_{CURRENT_USER}_{timestamp}"
    dirs = {
        'output': output_dir,
        'checkpoints': os.path.join(output_dir, "checkpoints"),
        'visualizations': os.path.join(output_dir, "visualizations"),
    }
    
    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(dirs['output'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Get transforms
    train_transform, val_transform, tta_transforms = get_transforms(config['img_size'])
    
    # Initialize training dataset
    try:
        logger.info(f"Loading training dataset from {paths['images_dir']}...")
        train_dataset = StenosisDataset(
            images_dir=paths['images_dir'],
            annotations_path=paths['annotations_path'],
            transform=train_transform,
            debug=True
        )
        
        logger.info(f"Training dataset loaded with {len(train_dataset)} samples")
        logger.info("Debugging training dataset...")
        train_dataset.debug_dataset(num_samples=2, save_dir=dirs['visualizations'])
        
        # Check if separate validation set exists
        use_separate_val = os.path.exists(paths['val_images_dir']) and os.path.exists(paths['val_annotations_path'])
        
        if use_separate_val:
            logger.info(f"Loading separate validation dataset from {paths['val_images_dir']}...")
            val_dataset = StenosisDataset(
                images_dir=paths['val_images_dir'],
                annotations_path=paths['val_annotations_path'],
                transform=val_transform,
                debug=True
            )
            logger.info(f"Validation dataset loaded with {len(val_dataset)} samples")
            logger.info("Debugging validation dataset...")
            val_dataset.debug_dataset(num_samples=2, save_dir=dirs['visualizations'])
            
            # Create data loaders directly
            val_indices = list(range(len(val_dataset)))
            val_dataset = Subset(val_dataset, val_indices)
        else:
            # Split the training dataset for validation
            train_size = int(config['train_split'] * len(train_dataset))
            val_size = len(train_dataset) - train_size
            
            indices = list(range(len(train_dataset)))
            
            if config['balanced_sampling']:
                # Ensure balanced class distribution in both train and val
                stenosis_indices = train_dataset.stenosis_indices
                normal_indices = train_dataset.normal_indices
                
                # Shuffle indices
                random.shuffle(stenosis_indices)
                random.shuffle(normal_indices)
                
                # Split for each class
                train_stenosis = stenosis_indices[:int(len(stenosis_indices) * config['train_split'])]
                val_stenosis = stenosis_indices[int(len(stenosis_indices) * config['train_split']):]
                
                train_normal = normal_indices[:int(len(normal_indices) * config['train_split'])]
                val_normal = normal_indices[int(len(normal_indices) * config['train_split']):]
                
                # Combine
                train_indices = train_stenosis + train_normal
                val_indices = val_stenosis + val_normal
                
                # Shuffle again
                random.shuffle(train_indices)
                random.shuffle(val_indices)
            else:
                # Regular random split
                random.shuffle(indices)
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
            
            # Create validation dataset with validation transform
            val_dataset = StenosisDataset(
                images_dir=paths['images_dir'],
                annotations_path=paths['annotations_path'],
                transform=val_transform,
                debug=False
            )
            val_dataset = Subset(val_dataset, val_indices)
        
        # Create train subset
        train_dataset = Subset(train_dataset, 
                               train_indices if not use_separate_val else list(range(len(train_dataset))))
        
        logger.info(f"Created train set with {len(train_dataset)} samples and val set with {len(val_dataset)} samples")
        
        # Calculate class weights for loss function
        if config['use_class_weights']:
            if use_separate_val:
                # Use all training data for weights
                train_labels = [train_dataset.dataset.stenosis_labels.get(
                    train_dataset.dataset.image_ids[idx], 0) for idx in range(len(train_dataset.dataset))]
            else:
                # Use only training subset
                train_labels = [train_dataset.dataset.stenosis_labels.get(
                    train_dataset.dataset.image_ids[idx], 0) for idx in train_indices]
            
            # Count classes
            class_counts = {}
            for label in train_labels:
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1
            
            # Calculate weights (inverse of frequency)
            class_weights = {}
            samples = sum(class_counts.values())
            for c in class_counts:
                class_weights[c] = samples / (len(class_counts) * class_counts[c]) if class_counts[c] > 0 else 1.0
            
            logger.info(f"Class distribution in training set: {class_counts}")
            logger.info(f"Class weights: {class_weights}")
        else:
            class_weights = {0: 1.0, 1: 1.0}
        
        # Create samplers for balanced batches
        if config['balanced_sampling']:
            if use_separate_val:
                train_labels = [train_dataset.dataset.stenosis_labels.get(
                    train_dataset.dataset.image_ids[idx], 0) for idx in range(len(train_dataset.dataset))]
            else:
                train_labels = [train_dataset.dataset.stenosis_labels.get(
                    train_dataset.dataset.image_ids[idx], 0) for idx in train_indices]
            
            weights = [class_weights[label] for label in train_labels]
            sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
        else:
            sampler = None
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=config['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
        
    except Exception as e:
        logger.error(f"Error setting up datasets: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = ModifiedResNet(num_classes=2, pretrained=True).to(device)
    logger.info(f"Initialized model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss function
    if config['use_focal_loss']:
        criterion = FocalLoss(alpha=0.75, gamma=2.0)  # Alpha favors minority class (stenosis)
        logger.info("Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([class_weights.get(0, 1.0), class_weights.get(1, 1.0)]).to(device)
        )
        logger.info("Using Weighted Cross Entropy Loss")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Learning rate scheduler - using the compatible version
    scheduler = create_compatible_scheduler(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # Training variables
    best_metric = 0.0  # Best F1
    best_epoch = 0
    early_stop_counter = 0
    
    # Training history
    history = {
        'train': {metric: [] for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']},
        'val': {metric: [] for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']}
    }
    
    # Training loop
    logger.info(f"Starting training for {config['num_epochs']} epochs")
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Training phase
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, criterion, device, None
        )
        
        # Validation phase
        val_metrics, val_labels, val_preds, val_probs, val_img_ids = validate(
            model, val_dataloader, criterion, device
        )
        
        # Update scheduler with manual logging of learning rate changes
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['f1'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log if learning rate changed
        if current_lr != prev_lr:
            logger.info(f"Learning rate changed from {prev_lr:.6f} to {current_lr:.6f}")
        
        # Log metrics
        for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
            if metric in train_metrics:
                history['train'][metric].append(train_metrics[metric])
            if metric in val_metrics:
                history['val'][metric].append(val_metrics[metric])
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{config['num_epochs']} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )
        
        # Log confusion matrix
        conf_mat = val_metrics['confusion_matrix']
        logger.info(f"Confusion matrix: [[TN={conf_mat[0,0]}, FP={conf_mat[0,1]}], [FN={conf_mat[1,0]}, TP={conf_mat[1,1]}]]")
        
        # Check for improvement
        current_metric = val_metrics['f1']
        
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            early_stop_counter = 0
            
            # Save best model
            checkpoint_path = os.path.join(dirs['checkpoints'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {
                    'accuracy': val_metrics['accuracy'],
                    'precision': val_metrics['precision'],
                    'recall': val_metrics['recall'],
                    'f1': val_metrics['f1']
                },
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved best model with F1={best_metric:.4f} to {checkpoint_path}")
        else:
            early_stop_counter += 1
            logger.info(f"No improvement for {early_stop_counter} epochs (best F1={best_metric:.4f} at epoch {best_epoch+1})")
            
            if early_stop_counter >= config['patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Visualize predictions and plot curves
        if (epoch + 1) % 3 == 0 or epoch == 0 or epoch == config['num_epochs'] - 1 or early_stop_counter >= config['patience']:
            try:
                vis_path = os.path.join(dirs['visualizations'], f'predictions_epoch{epoch+1}.png')
                visualize_predictions(model, val_dataloader, device, num_samples=8, save_path=vis_path)
                
                curves_path = os.path.join(dirs['visualizations'], f'curves_epoch{epoch+1}.png')
                plot_training_curves(history['train'], history['val'], curves_path)
            except Exception as e:
                logger.error(f"Error during visualization: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Training complete
    logger.info(f"Training completed. Best F1: {best_metric:.4f} at epoch {best_epoch+1}")
    
    # Load best model for final evaluation
    best_model_path = os.path.join(dirs['checkpoints'], 'best_model.pth')
    if os.path.exists(best_model_path):
        # Load model
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        logger.info("Performing final evaluation...")
        val_metrics, val_labels, val_preds, val_probs, val_img_ids = validate(
            model, val_dataloader, criterion, device
        )
        
        logger.info("Final evaluation metrics:")
        for metric, value in val_metrics.items():
            if metric != 'confusion_matrix':
                logger.info(f"- {metric}: {value:.4f}")
        
        conf_mat = val_metrics['confusion_matrix']
        logger.info(f"Final confusion matrix: [[TN={conf_mat[0,0]}, FP={conf_mat[0,1]}], [FN={conf_mat[1,0]}, TP={conf_mat[1,1]}]]")
        
        # Save detailed predictions for analysis
        predictions = []
        for img_id, label, pred, prob in zip(val_img_ids, val_labels, val_preds, val_probs):
            predictions.append({
                'image_id': int(img_id),
                'true_label': int(label),
                'predicted_label': int(pred),
                'probability': float(prob),
                'correct': int(label) == int(pred)
            })
        
        with open(os.path.join(dirs['output'], 'predictions.json'), 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Create detailed final visualization
        try:
            final_vis_path = os.path.join(dirs['visualizations'], 'final_predictions.png')
            visualize_predictions(model, val_dataloader, device, num_samples=16, save_path=final_vis_path)
        except Exception as e:
            logger.error(f"Error during final visualization: {e}")
    
    # Save training history
    with open(os.path.join(dirs['output'], 'training_history.json'), 'w') as f:
        # Convert numpy values to Python native types
        for split in ['train', 'val']:
            for metric in history[split]:
                history[split][metric] = [float(x) for x in history[split][metric]]
        json.dump(history, f, indent=2)
    
    # Save final training curves
    final_curves_path = os.path.join(dirs['visualizations'], 'final_training_curves.png')
    plot_training_curves(history['train'], history['val'], final_curves_path)
    
    return model, history


if __name__ == "__main__":
    logger.info(f"Script started by {CURRENT_USER} at {CURRENT_DATETIME}")
    try:
        model, history = main()
        if model is not None:
            logger.info(f"Script completed successfully. Best model saved in the output directory.")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        logger.error(traceback.format_exc())