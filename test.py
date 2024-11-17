import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# 1. 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 입력 크기에 맞게 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. 사용자 정의 데이터셋 클래스
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 흐릿한 이미지 폴더
        blurred_dir = os.path.join(root_dir, '0')
        # 선명한 이미지 폴더
        sharp_dir = os.path.join(root_dir, '1')
        shallowFocus_dir = os.path.join(root_dir, '2')

        # 각 폴더의 이미지를 리스트에 추가하고, 라벨도 지정
        for img in os.listdir(blurred_dir):
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                self.images.append(os.path.join(blurred_dir, img))
                self.labels.append(0)  # 흐릿한 이미지 라벨: 0

        for img in os.listdir(sharp_dir):
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                self.images.append(os.path.join(sharp_dir, img))
                self.labels.append(1)  # 선명한 이미지 라벨: 1

        for img in os.listdir(shallowFocus_dir):
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                self.images.append(os.path.join(shallowFocus_dir, img))
                self.labels.append(2)  # 얕은 심도 이미지 라벨: 2

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 3. ResNet 모델 설정 (전이 학습)
weights_path = "/Users/choizeus/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
model = models.resnet50()
model.load_state_dict(torch.load(weights_path, map_location='mps'), strict=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)  # 세 가지 분류용으로 마지막 층 조정

# 이전에 저장한 가중치가 있을 경우 불러오기
saved_weights_path = '/Users/choizeus/project/isSharp/blur_sharp_model_epoch_334.pth'
if os.path.exists(saved_weights_path):
    model.load_state_dict(torch.load(saved_weights_path, map_location='mps'))
    print(f"이전 학습된 가중치를 {saved_weights_path}에서 불러왔습니다.")

# 4. 학습 설정
device = torch.device("mps")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 데이터셋 및 데이터로더 생성
image_dir = '/Users/choizeus/Pictures/Photo/train/model'  # 최상위 이미지 디렉토리
dataset = ImageDataset(image_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 6. 에폭마다 가중치 저장을 포함한 학습 함수
def train_model(model, criterion, optimizer, data_loader, num_epochs=700):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.8f}")

        # 매 에폭 종료 시 가중치 저장
        epoch_weights_path = f'blur_sharp_model_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), epoch_weights_path)
        print(f"에폭 {epoch + 1} 가중치가 {epoch_weights_path}에 저장되었습니다.")

    print("학습 완료")

# 8. 모델 학습
train_model(model, criterion, optimizer, data_loader)
