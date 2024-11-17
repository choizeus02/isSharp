import argparse
import datetime
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
from pathlib import Path
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import ModelEma
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import build_dataset  # 데이터셋 구성 함수
from engine_finetune import train_one_epoch, evaluate  # 학습 및 평가 함수
from utils import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('3-Class ConvNeXt Fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=150, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay')
    parser.add_argument('--model_name', default='convnext_tiny', type=str, help='ConvNeXt model variant')
    parser.add_argument('--num_classes', default=3, type=int, help='Number of output classes (3)')
    parser.add_argument('--device', default='mps', help='Device to use for training/testing')
    parser.add_argument('--output_dir', default='./output', help='Output directory for saving models')
    parser.add_argument('--data_path', default='/Users/choizeus/Pictures/Photo/', type=str, help='Dataset path')
    parser.add_argument('--resume', default='./output/checkpoint_epoch_66.pth', help='Path to checkpoint to resume training')
    parser.add_argument('--model_ema', action='store_true', default=False, help='Use EMA for the model')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use AMP for training')
    parser.add_argument('--update_freq', default=1, type=int, help='Gradient accumulation steps')
    return parser


def main(args):
    device = torch.device(args.device)
    cudnn.benchmark = True
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"[INFO] Initializing model: {args.model_name}")
    model = create_model(args.model_name, pretrained=True, num_classes=args.num_classes)
    model.to(device)

    start_epoch = 0
    checkpoint = None  # checkpoint 변수를 먼저 초기화
    # 체크포인트 로드 시도
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"[INFO] Resumed training from epoch {start_epoch}")
        except FileNotFoundError:
            print(f"[WARNING] Checkpoint file '{args.resume}' not found. Starting training from scratch.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    print("[INFO] Building dataset")
    dataset_train, dataset_val = build_dataset(args.data_path, transform=transform)
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("[INFO] Setting up optimizer and criterion")
    criterion = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 옵티마이저 상태도 체크포인트에서 불러오기
    if checkpoint is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    model_ema = ModelEma(model, decay=0.9999) if args.model_ema else None
    loss_scaler = NativeScaler()

    print("[INFO] Starting training loop")
    for epoch in range(start_epoch, args.epochs):  # start_epoch부터 시작하도록 수정
        print(f"\n[INFO] Epoch {epoch + 1}/{args.epochs} - Training")
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            model_ema=model_ema,
            args=args,
        )
        print(f"[INFO] Training statistics: {train_stats}")

        if data_loader_val is not None:
            print(f"[INFO] Epoch {epoch + 1}/{args.epochs} - Validation")
            val_stats = evaluate(data_loader_val, model, device)
            print(f"[INFO] Validation statistics: {val_stats}")

        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"[INFO] Saved checkpoint to {checkpoint_path}")

    print("[INFO] Training complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt Fine-tuning for 3-Class Model', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
