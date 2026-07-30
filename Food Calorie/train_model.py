import os
import pandas as pd
import torch
import timm
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# ================= CONFIG =================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_FILE = "dataset.csv"
IMAGES_FOLDER = "static/images"  # base folder containing class folders

# ================= DATASET =================
class RecipeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.samples = []
        for _, row in df.iterrows():
            folder = os.path.join(IMAGES_FOLDER, row["image_folder_name"])
            if os.path.exists(folder):
                for img_name in os.listdir(folder):
                    img_path = os.path.join(folder, img_name)
                    self.samples.append((img_path, row["encoded_label"]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= LOAD CSV & LABEL ENCODER =================
df = pd.read_csv(CSV_FILE)
le = LabelEncoder()
df["encoded_label"] = le.fit_transform(df["image_class_label"])
NUM_CLASSES = df["image_class_label"].nunique()
pickle.dump(le, open("label_encoder.pkl", "wb"))

# ================= DATA LOADER =================
dataset = RecipeDataset(df, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ================= MODEL =================
# Create model with correct num_classes directly
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# Freeze backbone, train only head
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# ================= TRAIN SETUP =================
optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# ================= TRAIN LOOP =================
print(f"Starting training on {len(dataset)} images with {NUM_CLASSES} classes...")
model.train()

for epoch in range(EPOCHS):
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)  # already correct shape: (B, NUM_CLASSES)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(loader)}], Loss: {running_loss/10:.4f}")
            running_loss = 0.0

# ================= SAVE MODEL =================
torch.save(model.state_dict(), "model_swin.pth")
print("✅ Training finished! Model saved as 'model_swin.pth' and label_encoder.pkl saved.")
