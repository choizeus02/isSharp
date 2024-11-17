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

        # 하위 디렉토리 탐색 및 이미지 파일과 라벨 추가
        for subdir, _, files in os.walk(root_dir):
            for img in files:
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # 파일명에 '_0', '_1', '_2'가 있는지 확인 후 라벨 추출
                    if '_' in img:
                        try:
                            label = int(img.split('_')[-1][0])
                            self.images.append(os.path.join(subdir, img))
                            self.labels.append(label)
                        except ValueError:
                            print(f"Warning: 파일 '{img}'에서 라벨을 추출할 수 없습니다.")
                    else:
                        print(f"Warning: 파일명 '{img}'에 라벨 형식이 포함되어 있지 않습니다.")

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
model = models.convnext_base(pretrained=True)  # ConvNeXt 모델 불러오기 (사전 학습된 가중치 사용)
num_features = model.classifier[2].in_features  # ConvNeXt의 마지막 레이어 입력 특징 수
model.classifier[2] = nn.Linear(num_features, 3)  # 마지막 레이어를 삼진 분류로 변경

# 이전에 저장한 가중치가 있을 경우 불러오기
saved_weights_path = 'blur_sharp_model.pth'

# 4. 학습 설정
device = torch.device("mps")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 데이터셋 및 데이터로더 생성
image_dir = '/Users/choizeus/Pictures/Photo/train'  # 최상위 이미지 디렉토리
dataset = ImageDataset(image_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)


# 6. 학습 함수
def train_model(model, criterion, optimizer, data_loader, num_epochs=10):
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
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print("학습 완료")


# 7. 학습된 모델 저장
def save_model(model, path='/Users/choizeus/project/isSharp/blur_sharp_model.pth'):
    torch.save(model.state_dict(), path)
    print(f"모델이 {path}에 저장되었습니다.")


# 8. 모델 학습
train_model(model, criterion, optimizer, data_loader)
save_model(model, saved_weights_path)
