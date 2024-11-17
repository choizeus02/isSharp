import torch
from timm.models import create_model
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import CustomImageDataset  # CustomImageDataset 클래스
import argparse
import os
from PIL import Image

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt Testing', add_help=False)
    parser.add_argument('--model_name', default='convnext_tiny', type=str, help='ConvNeXt 모델 변형 선택')
    parser.add_argument('--num_classes', default=3, type=int, help='출력 클래스 수')
    parser.add_argument('--device', default='mps', help='테스트에 사용할 장치')
    parser.add_argument('--checkpoint_path',default='/Users/choizeus/project/isSharp/output/checkpoint_epoch_63.pth' , help='테스트할 체크포인트 경로')
    parser.add_argument('--data_path', default='/Users/choizeus/Pictures/Photo/train/FUJI/2024/1:26 (삿포로)/3일차/1', type=str, help='검증 데이터셋 경로')
    parser.add_argument('--batch_size', default=32, type=int, help='배치 크기')
    return parser

def main(args):
    device = torch.device(args.device)

    # 모델 초기화
    print(f"[INFO] Loading model: {args.model_name}")
    model = create_model(args.model_name, pretrained=True, num_classes=args.num_classes)
    model.to(device)

    # 체크포인트 로드
    print(f"[INFO] Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 평가 모드로 설정

    # 데이터 변환 설정
    print("[INFO] Setting up data transformations")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 데이터셋 준비
    print("[INFO] Preparing dataset")
    dataset_val = CustomImageDataset(root_dir=args.data_path, transform=transform)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 모델 평가
    print("[INFO] Starting evaluation")
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader_val):
            if images is None:  # None 데이터를 건너뜁니다.
                continue
            images = images.to(device)
            labels = labels.to(device)

            # 예측
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # 이미지별 결과 출력
            for j in range(images.size(0)):
                img_index = i * args.batch_size + j  # 이미지 인덱스
                file_name = dataset_val.images[img_index]  # 파일 이름 가져오기
                actual_label = labels[j].item()
                predicted_label = preds[j].item()
                print(f"Image: {file_name} | Predicted: {predicted_label}, Actual: {actual_label}")

            # 예측과 실제 라벨 비교
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

    # 정답률 계산
    accuracy = correct_predictions / total_predictions * 100
    print(f"[INFO] Total predictions: {total_predictions}, Correct predictions: {correct_predictions}")
    print(f"[INFO] Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt Model Testing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
