import torchvision
from torchvision import transforms
import numpy as np
import torch
import json
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the pretrained model
pretrained_model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
pretrained_model = pretrained_model.to(device)
pretrained_model.eval()

# Define normalization parameters
mean_norms = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std_norms = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Set up transforms
plain_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_norms, std=std_norms)
])

# Load dataset
dataset_path = "./TestDataSet"
dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=plain_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load label mapping
with open('TestDataSet/labels_list.json', 'r') as f:
    label_list = json.load(f)

# Build mapping from ImageFolder class idx to ImageNet idx
folder_to_idx = dataset.class_to_idx  
imagenet_indices = [int(label.split(':')[0]) for label in label_list]
idx_to_imagenet = {v: imagenet_indices[i] for i, (k, v) in enumerate(sorted(folder_to_idx.items()))}

# Evaluation function
def evaluate_model(model, data_loader, mapping_dict=None):
    """Evaluate model performance."""
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets_np = targets.cpu().numpy()
            outputs = model(images)
            
            # Get top-5 predictions
            _, pred_top5 = outputs.topk(5, dim=1)
            pred_top5 = pred_top5.cpu().numpy()
            
            # Map ground truth if mapping provided
            if mapping_dict:
                gt_imagenet = [mapping_dict[t] for t in targets_np]
            else:
                gt_imagenet = targets_np
            
            # Top-1 accuracy
            top1_preds = pred_top5[:, 0]
            top1_correct += sum([p == gt for p, gt in zip(top1_preds, gt_imagenet)])
            
            # Top-5 accuracy
            for i, gt in enumerate(gt_imagenet):
                if gt in pred_top5[i]:
                    top5_correct += 1
            
            total += len(targets)
    
    return top1_correct / total, top5_correct / total

# Evaluate original accuracy
print("\n=== Task 1: Evaluating Original Model ===")
top1_acc, top5_acc = evaluate_model(pretrained_model, dataloader, idx_to_imagenet)
print(f"Original Top-1 Accuracy: {top1_acc:.4f}")
print(f"Original Top-5 Accuracy: {top5_acc:.4f}")

# Utility functions
def denormalize(tensor, mean, std):
    """Convert normalized tensor back to [0,1] range."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def compute_linf_distance(original, perturbed):
    """Compute L∞ distance between original and perturbed images."""
    # Denormalize both to [0,1] space for fair comparison
    original_denorm = denormalize(original.clone().cpu(), mean_norms, std_norms)
    perturbed_denorm = denormalize(perturbed.clone().cpu(), mean_norms, std_norms)
    
    # Compute maximum absolute difference
    linf_dist = (original_denorm - perturbed_denorm).abs().max().item()
    return linf_dist

# Task 2: Implement FGSM exactly as described in the assignment
def fgsm_attack(model, image, target, epsilon=0.02):
    """
    A simpler FGSM implementation that avoids view issues and ensures
    exactly epsilon L∞ distance.
    """
    # Ensure correct data type
    image = image.clone().detach().to(device).float()
    image.requires_grad = True
    
    # Forward pass
    output = model(image)
    target_tensor = torch.tensor([target], dtype=torch.long).to(device)
    loss = torch.nn.functional.cross_entropy(output, target_tensor)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Get gradient sign
    data_grad = image.grad.data
    sign_data_grad = data_grad.sign()
    
    # Create perturbation
    # This approach works in normalized space
    # For an L∞ distance of epsilon in [0,1] image space,
    # we need to multiply by epsilon/std for each channel
    perturbation = torch.zeros_like(image)
    for c in range(3):
        # Scale epsilon by 1/std for each channel 
        perturbation[:, c:c+1] = sign_data_grad[:, c:c+1] * (epsilon / std_norms[c])
    
    # Apply perturbation
    perturbed_image = image + perturbation
    
    # Ensure valid range for the model
    mean_tensor = torch.tensor(mean_norms, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std_norms, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    mins = (torch.tensor(0.0, dtype=torch.float32, device=device) - mean_tensor) / std_tensor
    maxs = (torch.tensor(1.0, dtype=torch.float32, device=device) - mean_tensor) / std_tensor
    perturbed_image = torch.clamp(perturbed_image, mins, maxs)
    
    return perturbed_image.detach()
# Create directory for adversarial examples
adv_dir = 'AdversarialTestSet1'
os.makedirs(adv_dir, exist_ok=True)

# Ensure class subfolders exist
for class_name in dataset.classes:
    os.makedirs(os.path.join(adv_dir, class_name), exist_ok=True)

# Attack parameters
epsilon = 0.02  # As specified in the assignment
successful_examples = []  # Store successful attack examples for visualization

# Process each image
print("\n=== Task 2: Creating Adversarial Examples with FGSM ===")
for i, (img, target) in enumerate(dataset):
    # Original image (add batch dimension)
    img_batch = img.unsqueeze(0).to(device).float()
    
    # Use the true label for the attack (not the model's prediction)
    true_imagenet_label = idx_to_imagenet[target]
    perturbed_img = fgsm_attack(pretrained_model, img_batch, true_imagenet_label, epsilon)
    
    # Get adversarial prediction
    with torch.no_grad():
        adv_output = pretrained_model(perturbed_img)
        _, adv_pred = torch.max(adv_output, 1)
        adv_pred_imagenet = adv_pred.item()
    
    # Check if attack succeeded
    attack_succeeded = (adv_pred_imagenet != true_imagenet_label)
    
    # Verify L∞ distance constraint
    linf_dist = compute_linf_distance(img_batch, perturbed_img)
    
    # Store successful examples for visualization (if needed)
    if attack_succeeded and len(successful_examples) < 5:
        orig_img_denorm = denormalize(img.clone(), mean_norms, std_norms)
        pert_img_denorm = denormalize(perturbed_img.squeeze(0).cpu(), mean_norms, std_norms)
        
        # Get class names
        orig_label_name = next((label.split(':', 1)[1].strip() 
                          for label in label_list 
                          if label.startswith(f"{true_imagenet_label}:")), "Unknown")
        
        adv_label_name = next((label.split(':', 1)[1].strip() 
                          for label in label_list 
                          if label.startswith(f"{adv_pred_imagenet}:")), "Unknown")
        
        # Store for visualization
        successful_examples.append((
            orig_label_name,
            orig_img_denorm,
            pert_img_denorm,
            adv_label_name,
            linf_dist
        ))
    
    # Save perturbed image
    perturbed_img_denorm = denormalize(perturbed_img.squeeze(0).cpu(), mean_norms, std_norms)
    
    # Get save path based on original path
    orig_path, _ = dataset.samples[i]
    class_folder = orig_path.split(os.sep)[-2]
    img_name = os.path.basename(orig_path)
    save_path = os.path.join(adv_dir, class_folder, img_name)
    
    # Save the image
    save_image(perturbed_img_denorm, save_path)
    
    if i % 50 == 0:
        print(f"Processed {i+1}/{len(dataset)} images. L∞ distance: {linf_dist:.6f}")

print(f"All adversarial images saved to {adv_dir}")

# Visualize successful examples
if successful_examples:
    plt.figure(figsize=(20, 10))
    
    for i, (orig_label, orig_img, adv_img, adv_label, linf_dist) in enumerate(successful_examples):
        # Original
        plt.subplot(2, len(successful_examples), i+1)
        plt.imshow(orig_img.permute(1, 2, 0).numpy())
        plt.title(f"Original: {orig_label}")
        plt.axis('off')
        
        # Adversarial
        plt.subplot(2, len(successful_examples), i+len(successful_examples)+1)
        plt.imshow(adv_img.permute(1, 2, 0).numpy())
        plt.title(f"Adversarial: {adv_label}\nL∞ dist: {linf_dist:.6f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('fgsm_examples.png')
    print("Visualization saved to fgsm_examples.png")
else:
    print("No successful attack examples found for visualization.")

# Load adversarial dataset for evaluation
adv_dataset = torchvision.datasets.ImageFolder(root=adv_dir, transform=plain_transforms)
adv_loader = DataLoader(adv_dataset, batch_size=32, shuffle=False)

# Build mapping for adversarial dataset
adv_folder_to_idx = adv_dataset.class_to_idx
adv_idx_to_imagenet = {v: imagenet_indices[i] for i, (k, v) in enumerate(sorted(adv_folder_to_idx.items()))}

# Evaluate on adversarial dataset
print("\n=== Evaluating on Adversarial Test Set 1 ===")
adv_top1_acc, adv_top5_acc = evaluate_model(pretrained_model, adv_loader, adv_idx_to_imagenet)
print(f"Adversarial Top-1 Accuracy: {adv_top1_acc:.4f}")
print(f"Adversarial Top-5 Accuracy: {adv_top5_acc:.4f}")

# Calculate accuracy drop
top1_drop = (top1_acc - adv_top1_acc) / top1_acc * 100
top5_drop = (top5_acc - adv_top5_acc) / top5_acc * 100
print(f"\nAccuracy Drop:")
print(f"  Top-1 Accuracy Drop: {top1_drop:.2f}%")
print(f"  Top-5 Accuracy Drop: {top5_drop:.2f}%")

# Task 3: Improved attacks - Momentum Iterative FGSM
def momentum_iterative_fgsm(model, image, target, epsilon=0.02, steps=10, alpha=0.004, decay=0.9):
    """
    Momentum Iterative Fast Gradient Sign Method.
    
    Args:
        model: The model to attack
        image: Input image
        target: True label
        epsilon: Maximum perturbation (L∞ norm)
        steps: Number of attack iterations
        alpha: Step size for each iteration (should be epsilon/steps * factor)
        decay: Momentum decay factor
    
    Returns:
        perturbed_image: Adversarial example
    """
    # Initialize
    image = image.clone().detach().to(device).float()
    perturbed_image = image.clone()
    momentum_grad = torch.zeros_like(image).to(device)
    
    # Calculate bounds for valid normalized images
    mean_tensor = torch.tensor(mean_norms, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std_norms, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    mins = (torch.tensor(0.0, dtype=torch.float32, device=device) - mean_tensor) / std_tensor
    maxs = (torch.tensor(1.0, dtype=torch.float32, device=device) - mean_tensor) / std_tensor
    
    # Calculate channel-specific epsilon values for proper L∞ constraint
    epsilon_tensor = torch.tensor([epsilon / s for s in std_norms], 
                                 dtype=torch.float32, device=device).view(1, 3, 1, 1)
    alpha_tensor = torch.tensor([alpha / s for s in std_norms],
                               dtype=torch.float32, device=device).view(1, 3, 1, 1)
    
    # Iteratively apply the attack
    for i in range(steps):
        # Enable gradients
        perturbed_image.requires_grad = True
        
        # Forward pass
        output = model(perturbed_image)
        target_tensor = torch.tensor([target], dtype=torch.long).to(device)
        
        # Cross-entropy loss - for untargeted attack, we want to maximize this
        loss = torch.nn.functional.cross_entropy(output, target_tensor)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Get gradients and update momentum
        grad = perturbed_image.grad.data
        momentum_grad = decay * momentum_grad + grad / grad.norm(1)
        
        # Update image with momentum-enhanced gradient
        with torch.no_grad():
            grad_sign = momentum_grad.sign()
            perturbed_image = perturbed_image + alpha_tensor * grad_sign
            
            # Project back to epsilon-ball around original image
            delta = perturbed_image - image
            # Project channel-wise to maintain proper L∞ distance
            for c in range(3):
                delta[:, c:c+1] = torch.clamp(delta[:, c:c+1], 
                                             -epsilon_tensor[:, c:c+1], 
                                             epsilon_tensor[:, c:c+1])
            
            perturbed_image = image + delta
            
            # Ensure the result is within valid range
            perturbed_image = torch.clamp(perturbed_image, mins, maxs)
    
    return perturbed_image.detach()

# Create directory for improved adversarial examples
adv_dir2 = 'AdversarialTestSet2'
os.makedirs(adv_dir2, exist_ok=True)

# Ensure class subfolders exist
for class_name in dataset.classes:
    os.makedirs(os.path.join(adv_dir2, class_name), exist_ok=True)

# Improved attack parameters
epsilon = 0.02  # Same as Task 2
steps = 10      # Number of iterations 
alpha = 0.004   # Step size (0.2 * epsilon / steps) - using smaller steps for better convergence
decay = 0.9     # Momentum decay factor
successful_examples = []  # Store successful examples for visualization

# Process each image
print("\n=== Task 3: Creating Improved Adversarial Examples with MI-FGSM ===")
for i, (img, target) in enumerate(dataset):
    # Original image (add batch dimension)
    img_batch = img.unsqueeze(0).to(device).float()
    
    # Get true label
    true_imagenet_label = idx_to_imagenet[target]
    
    # Apply momentum iterative FGSM attack
    perturbed_img = momentum_iterative_fgsm(
        pretrained_model, img_batch, true_imagenet_label, 
        epsilon, steps, alpha, decay
    )
    
    # Get adversarial prediction
    with torch.no_grad():
        adv_output = pretrained_model(perturbed_img)
        _, adv_pred = torch.max(adv_output, 1)
        adv_pred_imagenet = adv_pred.item()
    
    # Check if attack succeeded
    attack_succeeded = (adv_pred_imagenet != true_imagenet_label)
    
    # Verify L∞ distance constraint
    linf_dist = compute_linf_distance(img_batch, perturbed_img)
    
    # Store successful examples for visualization
    if attack_succeeded and len(successful_examples) < 5:
        orig_img_denorm = denormalize(img.clone(), mean_norms, std_norms)
        pert_img_denorm = denormalize(perturbed_img.squeeze(0).cpu(), mean_norms, std_norms)
        
        # Get class names
        orig_label_name = next((label.split(':', 1)[1].strip() 
                          for label in label_list 
                          if label.startswith(f"{true_imagenet_label}:")), "Unknown")
        
        adv_label_name = next((label.split(':', 1)[1].strip() 
                          for label in label_list 
                          if label.startswith(f"{adv_pred_imagenet}:")), "Unknown")
        
        # Store for visualization
        successful_examples.append((
            orig_label_name,
            orig_img_denorm,
            pert_img_denorm,
            adv_label_name,
            linf_dist
        ))
    
    # Save perturbed image
    perturbed_img_denorm = denormalize(perturbed_img.squeeze(0).cpu(), mean_norms, std_norms)
    
    # Get save path based on original path
    orig_path, _ = dataset.samples[i]
    class_folder = orig_path.split(os.sep)[-2]
    img_name = os.path.basename(orig_path)
    save_path = os.path.join(adv_dir2, class_folder, img_name)
    
    # Save the image
    save_image(perturbed_img_denorm, save_path)
    
    if i % 50 == 0:
        print(f"Processed {i+1}/{len(dataset)} images. L∞ distance: {linf_dist:.6f}")

print(f"All improved adversarial images saved to {adv_dir2}")

# Visualize successful examples
if successful_examples:
    plt.figure(figsize=(20, 10))
    
    for i, (orig_label, orig_img, adv_img, adv_label, linf_dist) in enumerate(successful_examples):
        # Original
        plt.subplot(2, len(successful_examples), i+1)
        plt.imshow(orig_img.permute(1, 2, 0).numpy())
        plt.title(f"Original: {orig_label}")
        plt.axis('off')
        
        # Adversarial
        plt.subplot(2, len(successful_examples), i+len(successful_examples)+1)
        plt.imshow(adv_img.permute(1, 2, 0).numpy())
        plt.title(f"Adversarial: {adv_label}\nL∞ dist: {linf_dist:.6f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mifgsm_examples.png')
    print("Visualization saved to mifgsm_examples.png")
else:
    print("No successful attack examples found for visualization.")

# Load improved adversarial dataset for evaluation
adv_dataset2 = torchvision.datasets.ImageFolder(root=adv_dir2, transform=plain_transforms)
adv_loader2 = DataLoader(adv_dataset2, batch_size=32, shuffle=False)

# Build mapping for improved adversarial dataset
adv_folder_to_idx2 = adv_dataset2.class_to_idx
adv_idx_to_imagenet2 = {v: imagenet_indices[i] for i, (k, v) in enumerate(sorted(adv_folder_to_idx2.items()))}

# Evaluate on improved adversarial dataset
print("\n=== Evaluating on Adversarial Test Set 2 (MI-FGSM) ===")
adv_top1_acc2, adv_top5_acc2 = evaluate_model(pretrained_model, adv_loader2, adv_idx_to_imagenet2)
print(f"MI-FGSM Adversarial Top-1 Accuracy: {adv_top1_acc2:.4f}")
print(f"MI-FGSM Adversarial Top-5 Accuracy: {adv_top5_acc2:.4f}")

# Calculate accuracy drop
top1_drop2 = (top1_acc - adv_top1_acc2) / top1_acc * 100
top5_drop2 = (top5_acc - adv_top5_acc2) / top5_acc * 100
print(f"\nAccuracy Drop with MI-FGSM:")
print(f"  Top-1 Accuracy Drop: {top1_drop2:.2f}%")
print(f"  Top-5 Accuracy Drop: {top5_drop2:.2f}%")

# Compare with FGSM (Task 2) results
print("\n=== Comparison of Attack Methods ===")
print(f"Original Top-1 Accuracy: {top1_acc:.4f}")
print(f"FGSM (Task 2) Top-1 Accuracy: {adv_top1_acc:.4f} (Drop: {top1_drop:.2f}%)")
print(f"MI-FGSM (Task 3) Top-1 Accuracy: {adv_top1_acc2:.4f} (Drop: {top1_drop2:.2f}%)")
print(f"Task 3 improvement over Task 2: {top1_drop2 - top1_drop:.2f}%")