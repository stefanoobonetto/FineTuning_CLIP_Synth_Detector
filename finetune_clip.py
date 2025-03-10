import os
import torch
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
import numpy as np

device = "mps"  

print("Loading CLIP model...")
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
print("CLIP model loaded successfully!")

AUGMENTATIONS = [
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
]
class SyntheticDataset(Dataset):
    def __init__(self, root_dir, processor=None, apply_augmentation=True):
        self.image_paths = []
        self.labels = []
        self.processor = processor
        self.video_to_augmentation = {}
        self.apply_augmentation = apply_augmentation
        
        self.scan_dataset(root_dir)
        print(f"Dataset loaded: {len(self.image_paths)} images (original + augmented) found")
        
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

                self.video_to_augmentation[video_folder] = random.choice(AUGMENTATIONS)
                frame_count = 0

                for frame in os.listdir(video_path):
                    frame_path = os.path.join(video_path, frame)
                    if frame_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append((frame_path, label, False))  # Original
                        self.image_paths.append((frame_path, label, True))   # Augmented
                        frame_count += 2  # Counting both original and augmented
                
                print(f"  - Added {frame_count} samples (original + augmented) from {video_folder} ({class_name})")

    def extract_video_id(self, path):
        return os.path.basename(os.path.dirname(path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, label, augmented = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        if augmented:
            video_id = self.extract_video_id(image_path)
            augmentation = self.video_to_augmentation[video_id]
            image = augmentation(image)
        
        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            pixel_values = inputs.pixel_values.squeeze(0)
            return pixel_values, label, image_path  
        
        return image, label, image_path
    
def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    paths = [item[2] for item in batch]
    return pixel_values, labels, paths

class CLIPClassifier(torch.nn.Module):
    def __init__(self, clip_model):
        super(CLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.fc = torch.nn.Linear(768, 2)  # real or fake

    def forward(self, pixel_values):
        with torch.no_grad():  
            # Extract vision features only
            vision_outputs = self.clip_model.vision_model(pixel_values)
            image_features = vision_outputs.pooler_output
        return self.fc(image_features)

def train(model, dataloader, epochs=5):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)  
    
    history = {
        'loss': [],
        'accuracy': []
    }

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, _) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        epoch_loss = total_loss/len(dataloader)
        epoch_accuracy = 100 * correct / total
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    return history

def test_model(model, test_loader):
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            for i in range(len(labels)):
                results.append({
                    'path': paths[i],
                    'true_label': labels[i].item(),
                    'predicted': predictions[i].item(),
                    'confidence': probabilities[i][predictions[i]].item(),
                    'real_prob': probabilities[i][0].item(),
                    'fake_prob': probabilities[i][1].item()
                })
    
    return results

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved to 'training_history.png'")

def main():
    print("Starting DeepFake detection program...")
    
    train_root = os.path.join("data", "train", "train_set_1")
    if not os.path.exists(train_root):
        print(f"Error: Data directory not found: {train_root}")
        print("Current working directory:", os.getcwd())
        print("Available directories:", os.listdir())
        return
    
    print(f"Data directory found: {train_root}")
    
    train_dataset = SyntheticDataset(train_root, processor=processor)
    
    if len(train_dataset) == 0:
        print("No images found in the dataset. Exiting.")
        return
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True,
        collate_fn=collate_fn
    )

    print("Creating CLIP classifier model...")
    model = CLIPClassifier(clip_model).to(device)
    print("Model created successfully!")

    model_path = "clip_deepfake_detector.pth"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Starting training...")
        history = train(model, train_loader, epochs=10)
        
        plot_training_history(history)
        
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print("Testing model on a few samples...")
    test_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    results = test_model(model, test_loader)
    
    print("\nTest Results:")
    print("-" * 80)
    print(f"{'Image Path':<50} | {'True':<5} | {'Pred':<5} | {'Conf':<10} | {'Real%':<10} | {'Fake%':<10}")
    print("-" * 80)
    
    for i, res in enumerate(results[:10]):  
        path_short = os.path.basename(os.path.dirname(res['path'])) + "/" + os.path.basename(res['path'])
        print(f"{path_short:<50} | {res['true_label']:<5} | {res['predicted']:<5} | {res['confidence']:.4f} | {res['real_prob']:.4f} | {res['fake_prob']:.4f}")
    
    correct = sum(1 for res in results if res['true_label'] == res['predicted'])
    accuracy = correct / len(results) * 100
    print("-" * 80)
    print(f"Test accuracy: {accuracy:.2f}% ({correct}/{len(results)})")
    
    print("\nProgram completed successfully!")

if __name__ == "__main__":
    main()