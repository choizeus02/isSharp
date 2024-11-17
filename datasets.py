from torch.utils.data import Dataset
import os
from PIL import Image

class CustomImageDataset(Dataset):from torch.utils.data import Dataset
import os
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # root_dir의 모든 하위 폴더를 순회하며 이미지 파일을 추가
        for dirpath, _, filenames in os.walk(root_dir):
            for file_name in filenames:
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(dirpath, file_name)
                    try:
                        # _ 뒤의 숫자를 라벨로 사용
                        label = int(file_name.split('_')[-1].split('.')[0])
                        self.images.append(file_path)
                        self.labels.append(label)
                    except ValueError:
                        print(f"파일 이름에서 라벨을 추출할 수 없습니다: {file_name}")
                        # 라벨이 추출되지 않는 파일을 건너뛰기 위해 continue 추가
                        continue

        # 오류 방지: 라벨 수가 이미지 수와 일치하는지 확인
        assert len(self.images) == len(self.labels), "이미지와 라벨의 수가 일치하지 않습니다."
        print(f"로드된 이미지 수: {len(self.images)}, 라벨 수: {len(self.labels)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # 파일을 열다가 발생하는 오류를 처리
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"File not found: {img_path}, skipping...")
            return None, None

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def build_dataset(data_path, transform=None):
    # 모든 하위 폴더를 탐색하여 CustomImageDataset 인스턴스 생성
    dataset_train = CustomImageDataset(root_dir=os.path.join(data_path, 'train'), transform=transform)
    dataset_val = CustomImageDataset(root_dir=os.path.join(data_path, 'val'), transform=transform)

    print(f"Number of classes = {len(set(dataset_train.labels))}")  # 고유 라벨 수 출력

    return dataset_train, dataset_val
