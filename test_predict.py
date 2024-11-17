import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import shutil

# 모델 및 데이터 전처리 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 모델 불러오기 함수
def load_model(path='/Users/choizeus/project/isSharp/blur_sharp_model_epoch_309.pth'):
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)  # 3분류로 변경
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"{path}에서 모델을 불러왔습니다.")
    return model

# 이미지 예측 함수
def predict_image(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # 예측 결과에 따라 라벨 반환
    if predicted.item() == 0:
        return "흐릿한 이미지"
    elif predicted.item() == 1:
        return "선명한 이미지"
    else:
        return "얕은 심도 이미지"

# 폴더 내 모든 이미지 파일 예측 및 이동 함수
def predict_images_in_folder(folder_path, model, transform, blur_folder, sharp_folder, shallow_focus_folder):
    # 각 폴더가 없으면 생성
    os.makedirs(blur_folder, exist_ok=True)
    os.makedirs(sharp_folder, exist_ok=True)
    os.makedirs(shallow_focus_folder, exist_ok=True)

    # 폴더 내 파일 목록을 순회
    for file_name in os.listdir(folder_path):
        # 이미지 파일 확장자 필터링
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, file_name)
            result = predict_image(image_path, model, transform)
            print(f"{file_name}: {result}")

            # 예측 결과에 따라 파일 이동
            if result == "선명한 이미지":
                shutil.move(image_path, os.path.join(sharp_folder, file_name))
            elif result == "흐릿한 이미지":
                shutil.move(image_path, os.path.join(blur_folder, file_name))
            else:
                shutil.move(image_path, os.path.join(shallow_focus_folder, file_name))


def calculate_accuracy(blur_folder, sharp_folder, shallow_focus_folder):
    correct = 0
    total = 0

    # Helper function to check if a file has a specific suffix and increment counts
    def check_folder(folder_path, expected_label):
        nonlocal correct, total
        for file_name in os.listdir(folder_path):
            # Check if file matches the expected suffix
            if file_name.endswith(f"_{expected_label}.jpg") or file_name.endswith(f"_{expected_label}.jpeg") or file_name.endswith(f"_{expected_label}.png"):
                correct += 1
            total += 1

    # Check each folder with the expected label
    check_folder(blur_folder, "0")  # "0" expected in blur folder
    check_folder(sharp_folder, "1")  # "1" expected in sharp folder
    check_folder(shallow_focus_folder, "2")  # "2" expected in shallow focus folder

    # Calculate accuracy
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"총 이미지 개수: {total}")
    print(f"정답 개수: {correct}")
    print(f"정답률: {accuracy:.2f}%")

# 예측 실행
if __name__ == "__main__":
    folder_path = '/Users/choizeus/Pictures/Photo/train/model/test'  # 예측할 이미지가 들어 있는 폴더 경로
    blur_folder = '/Users/choizeus/Pictures/Photo/train/model/test0'  # 흐릿한 이미지를 이동할 폴더 경로
    sharp_folder = '/Users/choizeus/Pictures/Photo/train/model/test1'  # 선명한 이미지를 이동할 폴더 경로
    shallow_focus_folder = '/Users/choizeus/Pictures/Photo/train/model/test2'  # 얕은 심도 이미지를 이동할 폴더 경로

    # 모델 불러오기
    model = load_model('/Users/choizeus/project/isSharp/blur_sharp_model_epoch_309.pth')

    # 폴더 내 이미지 예측 수행 및 결과 출력, 파일 이동
    predict_images_in_folder(folder_path, model, transform, blur_folder, sharp_folder, shallow_focus_folder)


    calculate_accuracy(blur_folder, sharp_folder, shallow_focus_folder)

