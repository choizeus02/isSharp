
# resize 결과 확인기

import os
from PIL import Image
import torch
import torchvision.transforms as transforms

# 원본 이미지 불러오기
image_path = "/Users/choizeus/project/isSharp/test_images/image1.jpeg"  # 확인하고자 하는 이미지 경로 설정
image = Image.open(image_path).convert("RGB")

# 저장 폴더 설정
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# 변환 설정
transforms_list = {
    "resize_512x512": transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ]),
    "resize_then_center_crop_512x512": transforms.Compose([
        transforms.Resize(512),  # 짧은 변을 512로 맞추기
        transforms.CenterCrop((512, 512)),
        transforms.ToTensor()
    ]),
    "resize_then_pad_512x512": transforms.Compose([
        transforms.Resize(512),
        transforms.Pad((512, 512)),
        transforms.ToTensor()
    ]),
    "resize_550_then_random_resized_crop_512x512": transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomResizedCrop((512, 512)),
        transforms.ToTensor()
    ])
}

# 변환된 이미지 저장
for name, transform in transforms_list.items():
    transformed_image = transform(image)
    transformed_image = transforms.ToPILImage()(transformed_image)  # Tensor를 PIL 이미지로 변환

    save_path = os.path.join(output_folder, f"{name}.jpg")
    transformed_image.save(save_path)
    print(f"{name} 변환 결과가 {save_path}에 저장되었습니다.")
