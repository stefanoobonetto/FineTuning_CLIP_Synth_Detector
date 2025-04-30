import os
import torch
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading CLIP model...")
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
print("CLIP model loaded successfully!")

AUGMENTATIONS = [
    lambda img: img.convert("RGB").save("temp.jpg", "JPEG", quality=random.randint(50, 100)) or Image.open("temp.jpg"),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-10, 10))
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

class CLIPClassifier(torch.nn.Module):
    def __init__(self, clip_model):
        super(CLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.fc = torch.nn.Linear(768, 2)  # (real/fake)
    
    def forward(self, pixel_values):
        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(pixel_values)
            image_features = vision_outputs.pooler_output
        return self.fc(image_features)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(dataloader), 100 * correct / total
def train(model, train_dataloader, val_dataloader, epochs=5, patience=3):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)  
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for images, labels, _ in train_dataloader:
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
                
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        
        train_loss = total_loss / len(train_dataloader)
        train_accuracy = 100 * correct / total
        val_loss, val_accuracy = evaluate(model, val_dataloader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break

def main():
    train_dataset = SyntheticDataset("data/train/train_set_1", processor=processor)
    val_dataset = SyntheticDataset("data/val/val_set_1", processor=processor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    base_model = CLIPModel.from_pretrained(model_name).to(device)
    model = CLIPClassifier(base_model).to(device)
    
    train(model, train_loader, val_loader, epochs=10, patience=3)
    print("Training completed!")

if __name__ == "__main__":
    main()