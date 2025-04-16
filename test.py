import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, accuracy_score, confusion_matrix, classification_report

device = "mps"  # Change as needed (cuda, cpu, mps)

# Load the same model architecture
print("Loading CLIP model architecture...")
model_name = "openai/clip-vit-base-patch32"
base_model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Define the same dataset class
class TestDataset(Dataset):
    def __init__(self, root_dir, processor=None):
        self.image_paths = []
        self.labels = []
        self.processor = processor
        
        self.scan_dataset(root_dir)
        print(f"Test dataset loaded: {len(self.image_paths)} images found")
        
    def scan_dataset(self, root_dir):
        for label, class_name in enumerate(["0_real", "1_fake"]):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Path not found: {class_path}")
                continue

            print(f"Scanning {class_path}...")
            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)
                if not os.path.isdir(video_path):
                    continue

                frame_count = 0
                for frame in os.listdir(video_path):
                    frame_path = os.path.join(video_path, frame)
                    if frame_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(frame_path)
                        self.labels.append(label)
                        frame_count += 1
                
                print(f"  - Added {frame_count} samples from {video_folder} ({class_name})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            pixel_values = inputs.pixel_values.squeeze(0)
            return pixel_values, label, image_path
        
        return image, label, image_path

# Define the same classifier architecture
class CLIPClassifier(torch.nn.Module):
    def __init__(self, clip_model):
        super(CLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.fc = torch.nn.Linear(768, 2)  # Binary classification (real/fake)
    
    def forward(self, pixel_values):
        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(pixel_values)
            image_features = vision_outputs.pooler_output
        return self.fc(image_features)

def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    incorrect_predictions = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # For each batch, collect predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Track incorrect predictions
            for i, (pred, label) in enumerate(zip(predicted, labels)):
                if pred != label:
                    incorrect_predictions.append({
                        'path': paths[i],
                        'true': label.item(),
                        'pred': pred.item()
                    })
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=["Real", "Fake"])
    
    # Print results
    print(f"\nTest Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ["Real", "Fake"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to confusion matrix
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Save incorrect predictions for analysis
    if incorrect_predictions:
        print(f"\nIncorrect Predictions: {len(incorrect_predictions)}")
        with open('incorrect_predictions.txt', 'w') as f:
            for item in incorrect_predictions:
                f.write(f"Path: {item['path']}, True: {item['true']}, Predicted: {item['pred']}\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'incorrect_predictions': incorrect_predictions
    }

def main():
    # Load test data
    test_dataset = TestDataset("data/test/test_set_1", processor=processor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model with the same architecture
    model = CLIPClassifier(base_model).to(device)
    
    # Load the fine-tuned model weights
    model.load_state_dict(torch.load("best_model.pth"))
    print("Model loaded successfully!")
    
    # Test the model
    metrics = test_model(model, test_loader)
    
    # Optionally visualize some incorrect predictions
    if metrics['incorrect_predictions']:
        # Display some sample incorrect predictions (adjust the number as needed)
        num_samples = min(5, len(metrics['incorrect_predictions']))
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        
        for i in range(num_samples):
            sample = metrics['incorrect_predictions'][i]
            img = Image.open(sample['path']).convert('RGB')
            if num_samples > 1:
                axes[i].imshow(img)
                axes[i].set_title(f"True: {sample['true']}, Pred: {sample['pred']}")
                axes[i].axis('off')
            else:
                axes.imshow(img)
                axes.set_title(f"True: {sample['true']}, Pred: {sample['pred']}")
                axes.axis('off')
        
        plt.tight_layout()
        plt.savefig('incorrect_samples.png')
        plt.close()

if __name__ == "__main__":
    main()