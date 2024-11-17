# import os
# import shutil
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
#
# # 모델 및 데이터 전처리 로드
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# # 모델 불러오기 함수
# def load_model(path='blur_sharp_model_conv.pth'):
#     model = models.resnet50()
#     num_features = model.fc.in_features
#     model.fc = nn.Linear(num_features, 3)  # 세 개의 클래스로 수정
#     model.load_state_dict(torch.load(path, map_location=device))
#     model = model.to(device)
#     model.eval()
#     print(f"{path}에서 모델을 불러왔습니다.")
#     return model
#
# # 이미지 예측 함수
# def predict_image(image_path, model, transform):
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output, 1)
#
#     if predicted.item() == 0:
#         return "흐릿한 이미지"
#     elif predicted.item() == 1:
#         return "선명한 이미지"
#     elif predicted.item() == 2:
#         return "얕은 심도 이미지"
#
# # 폴더 내 모든 이미지 파일 예측 및 이동 함수
# def predict_images_in_folder(folder_path, model, transform, blur_folder, sharp_folder, shallow_focus_folder):
#     # 각 폴더가 없으면 생성
#     os.makedirs(blur_folder, exist_ok=True)
#     os.makedirs(sharp_folder, exist_ok=True)
#     os.makedirs(shallow_focus_folder, exist_ok=True)
#
#     # 폴더 내 파일 목록을 순회
#     for file_name in os.listdir(folder_path):
#         # 이미지 파일 확장자 필터링
#         if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
#             image_path = os.path.join(folder_path, file_name)
#             result = predict_image(image_path, model, transform)
#             print(f"{file_name}: {result}")
#
#             # 예측 결과에 따라 파일 이동
#             if result == "선명한 이미지":
#                 shutil.move(image_path, os.path.join(sharp_folder, file_name))
#             elif result == "흐릿한 이미지":
#                 shutil.move(image_path, os.path.join(blur_folder, file_name))
#             else:
#                 shutil.move(image_path, os.path.join(shallow_focus_folder, file_name))
#
# # 예측 실행
# if __name__ == "__main__":
#     folder_path = '/Users/choizeus/project/isSharp/test_images'  # 예측할 이미지가 들어 있는 폴더 경로
#     blur_folder = '/Users/choizeus/project/isSharp/blur_images'  # 흐릿한 이미지를 이동할 폴더 경로
#     sharp_folder = '/Users/choizeus/project/isSharp/sharp_images'  # 선명한 이미지를 이동할 폴더 경로
#     shallow_focus_folder = '/Users/choizeus/project/isSharp/shallow_focus_images'  # 얕은 심도 이미지를 이동할 폴더 경로
#
#     # 모델 불러오기
#     model = load_model('results/blur_sharp_model_conv.pth')
#
#     # 폴더 내 이미지 예측 수행 및 결과 출력, 파일 이동
#     predict_images_in_folder(folder_path, model, transform, blur_folder, sharp_folder, shallow_focus_folder)

import os
import shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 모델 및 데이터 전처리 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 모델 불러오기 함수
def load_model(path='blur_sharp_model_conv.pth'):
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
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

    return "선명한 이미지" if predicted.item() == 1 else "흐릿한 이미지"

# 폴더 내 모든 이미지 파일 예측 및 이동 함수
def predict_images_in_folder(folder_path, model, transform, blur_folder, sharp_folder):
    # 각 폴더가 없으면 생성
    os.makedirs(blur_folder, exist_ok=True)
    os.makedirs(sharp_folder, exist_ok=True)

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
            else:
                shutil.move(image_path, os.path.join(blur_folder, file_name))

# 예측 실행
if __name__ == "__main__":
    folder_path = '/Users/choizeus/project/isSharp/test_images'  # 예측할 이미지가 들어 있는 폴더 경로
    blur_folder = '/Users/choizeus/project/isSharp/blur_images'  # 흐릿한 이미지를 이동할 폴더 경로
    sharp_folder = '/Users/choizeus/project/isSharp/sharp_images'  # 선명한 이미지를 이동할 폴더 경로

    # 모델 불러오기
    model = load_model('../results/blur_sharp_model.pth')

    # 폴더 내 이미지 예측 수행 및 결과 출력, 파일 이동
    predict_images_in_folder(folder_path, model, transform, blur_folder, sharp_folder)
