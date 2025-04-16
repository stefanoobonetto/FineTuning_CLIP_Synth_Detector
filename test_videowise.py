import os
import torch
from PIL import Image
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, accuracy_score, confusion_matrix, classification_report

device = "mps"  

# Load the model architecture
print("Loading CLIP model architecture...")
model_name = "openai/clip-vit-base-patch32"
base_model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Define the dataset class for test data
class TestDataset(Dataset):
    def __init__(self, root_dir, processor=None):
        self.image_paths = []
        self.labels = []
        self.video_ids = []  # Track which video each frame belongs to
        self.video_info = {}  # Store information about each video
        self.processor = processor
        
        self.scan_dataset(root_dir)
        print(f"Test dataset loaded: {len(self.image_paths)} images found across {len(self.video_info)} videos")
        
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

                # Store information about this video
                self.video_info[video_folder] = {
                    'label': label,
                    'class': class_name,
                    'frame_count': 0,
                    'path': video_path
                }
                
                frame_count = 0
                for frame in os.listdir(video_path):
                    frame_path = os.path.join(video_path, frame)
                    if frame_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(frame_path)
                        self.labels.append(label)
                        self.video_ids.append(video_folder)
                        frame_count += 1
                
                self.video_info[video_folder]['frame_count'] = frame_count
                print(f"  - Added {frame_count} frames from {video_folder} ({class_name})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        video_id = self.video_ids[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            pixel_values = inputs.pixel_values.squeeze(0)
            return pixel_values, label, image_path, video_id
        
        return image, label, image_path, video_id

# Define the classifier architecture
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

def test_model_video_level(model, test_loader, test_dataset):
    model.eval()
    
    # Store predictions for each frame
    frame_predictions = {}
    
    # Dictionary to store predictions for each video
    video_predictions = defaultdict(list)
    
    with torch.no_grad():
        for images, labels, paths, video_ids in tqdm(test_loader, desc="Testing frames"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions by frame and by video
            for i, (pred, vid, path) in enumerate(zip(predicted, video_ids, paths)):
                frame_predictions[path] = {
                    'true': labels[i].item(),
                    'pred': pred.item()
                }
                video_predictions[vid].append(pred.item())
    
    # Aggregate video-level predictions based on majority vote
    video_level_preds = {}
    video_level_true = {}
    
    for video_id, predictions in video_predictions.items():
        # Count number of frames predicted as real (0) vs fake (1)
        real_count = predictions.count(0)
        fake_count = predictions.count(1)
        
        # Majority vote: if most frames are classified as real, the video is real
        video_level_preds[video_id] = 0 if real_count > fake_count else 1
        video_level_true[video_id] = test_dataset.video_info[video_id]['label']
    
    # Create lists for metrics calculation
    true_labels = list(video_level_true.values())
    predicted_labels = list(video_level_preds.values())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    class_report = classification_report(true_labels, predicted_labels, target_names=["Real", "Fake"])
    
    # Find incorrectly classified videos
    incorrect_videos = []
    for video_id in video_level_preds:
        if video_level_preds[video_id] != video_level_true[video_id]:
            incorrect_videos.append({
                'video_id': video_id,
                'true': video_level_true[video_id],
                'pred': video_level_preds[video_id],
                'real_count': video_predictions[video_id].count(0),
                'fake_count': video_predictions[video_id].count(1),
                'total_frames': len(video_predictions[video_id]),
                'path': test_dataset.video_info[video_id]['path']
            })
    
    # Print results
    print(f"\nVideo-Level Test Metrics:")
    print(f"Number of videos: {len(video_level_preds)}")
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
    plt.title('Video-Level Confusion Matrix')
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
    plt.savefig('video_level_confusion_matrix.png')
    
    # Save detailed predictions and analysis
    with open('video_level_predictions.txt', 'w') as f:
        f.write("Video ID,True Label,Predicted Label,Real Frames,Fake Frames,Total Frames,Accuracy\n")
        for video_id in sorted(video_level_preds.keys()):
            real_count = video_predictions[video_id].count(0)
            fake_count = video_predictions[video_id].count(1)
            total = len(video_predictions[video_id])
            correct = video_level_preds[video_id] == video_level_true[video_id]
            f.write(f"{video_id},{video_level_true[video_id]},{video_level_preds[video_id]},{real_count},{fake_count},{total},{correct}\n")
    
    # Save details of incorrectly classified videos
    if incorrect_videos:
        print(f"\nIncorrectly classified videos: {len(incorrect_videos)}")
        with open('incorrect_videos.txt', 'w') as f:
            for video in incorrect_videos:
                f.write(f"Video ID: {video['video_id']}\n")
                f.write(f"True label: {video['true']} ({'Real' if video['true'] == 0 else 'Fake'})\n")
                f.write(f"Predicted label: {video['pred']} ({'Real' if video['pred'] == 0 else 'Fake'})\n")
                f.write(f"Frame distribution: {video['real_count']} real / {video['fake_count']} fake out of {video['total_frames']} frames\n")
                f.write(f"Path: {video['path']}\n\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'incorrect_videos': incorrect_videos,
        'video_level_preds': video_level_preds,
        'video_level_true': video_level_true,
        'frame_predictions': frame_predictions
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
    
    # Test the model at video level
    metrics = test_model_video_level(model, test_loader, test_dataset)
    
    # Create visualization of prediction distribution for each video
    plt.figure(figsize=(12, 8))
    
    # Get list of videos and sort by true label then video ID for better visualization
    videos = list(metrics['video_level_true'].keys())
    videos.sort(key=lambda v: (metrics['video_level_true'][v], v))
    
    real_videos = [v for v in videos if metrics['video_level_true'][v] == 0]
    fake_videos = [v for v in videos if metrics['video_level_true'][v] == 1]
    
    # Number of videos
    n_videos = len(videos)
    
    # Create subplots - one for real videos, one for fake videos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot for real videos
    if real_videos:
        real_data = []
        real_labels = []
        for vid in real_videos:
            real_count = sum(1 for frame in metrics['frame_predictions'].values() if frame['pred'] == 0 and frame['true'] == 0)
            fake_count = sum(1 for frame in metrics['frame_predictions'].values() if frame['pred'] == 1 and frame['true'] == 0)
            total = real_count + fake_count
            real_pct = real_count / total * 100 if total > 0 else 0
            real_data.append(real_pct)
            real_labels.append(vid)

        real_colors = ['green' if metrics['video_level_preds'][v] == 0 else 'red' for v in real_videos]
        ax1.bar(range(len(real_videos)), real_data, color=real_colors)
        ax1.set_xticks(range(len(real_videos)))
        ax1.set_xticklabels(real_labels, rotation=90)
        ax1.set_ylabel('% Frames Classified as Real')
        ax1.set_title('Real Videos - Frame Classification Distribution')
        ax1.axhline(y=50, color='black', linestyle='-', alpha=0.3)  # 50% line

    # Plot for fake videos
    if fake_videos:
        fake_data = []
        fake_labels = []
        for vid in fake_videos:
            real_count = sum(1 for frame in metrics['frame_predictions'].values() if frame['pred'] == 0 and frame['true'] == 1)
            fake_count = sum(1 for frame in metrics['frame_predictions'].values() if frame['pred'] == 1 and frame['true'] == 1)
            total = real_count + fake_count
            fake_pct = fake_count / total * 100 if total > 0 else 0
            fake_data.append(fake_pct)
            fake_labels.append(vid)

        fake_colors = ['green' if metrics['video_level_preds'][v] == 1 else 'red' for v in fake_videos]
        ax2.bar(range(len(fake_videos)), fake_data, color=fake_colors)
        ax2.set_xticks(range(len(fake_videos)))
        ax2.set_xticklabels(fake_labels, rotation=90)
        ax2.set_ylabel('% Frames Classified as Fake')
        ax2.set_title('Fake Videos - Frame Classification Distribution')
        ax2.axhline(y=50, color='black', linestyle='-', alpha=0.3)  # 50% line

        plt.tight_layout()
        plt.savefig('video_classification_distribution.png')
        plt.close()

    print("\nResults and visualizations saved to disk.")
    print("- video_level_confusion_matrix.png: Confusion matrix for video-level classification")
    print("- video_classification_distribution.png: Distribution of frame classifications for each video")
    print("- video_level_predictions.txt: Detailed predictions for all videos")
    if metrics['incorrect_videos']:
        print("- incorrect_videos.txt: Details about incorrectly classified videos")

if __name__ == "__main__":
    main()