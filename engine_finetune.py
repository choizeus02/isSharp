import math
from typing import Iterable
import torch
from timm.utils import accuracy
import utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema=None, mixup_fn=None, log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    optimizer.zero_grad()
    use_amp = args.use_amp  # AMP 사용 여부 확인
    print(f"Starting training epoch {epoch+1}")

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        print(f"Processing batch {data_iter_step + 1}")
        samples = samples.to(device).contiguous().reshape(-1, *samples.shape[-3:])
        targets = targets.to(device).contiguous().reshape(-1)
        print(f"Samples shape after reshape: {samples.shape}, Targets shape after reshape: {targets.shape}")

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else:
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()
        print(f"Batch {data_iter_step + 1} Loss: {loss_value}")

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            break

        # # 손실 역전파 및 최적화 단계
        # if use_amp:
        #     print(f"test")
        #     loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters())
        # else:
        #     print(f"test2")
        #     loss.backward()
        #     print(f"test3")
        #     if (data_iter_step + 1) % args.update_freq == 0:
        #         optimizer.step()
        #         optimizer.zero_grad()
        print(f"Backpropagation and optimization done for batch {data_iter_step + 1}")

        # 정확도 계산 및 로깅
        class_acc = (output.max(-1)[-1] == targets).float().mean().item()
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        print(f"Batch {data_iter_step + 1} Accuracy: {class_acc * 100:.2f}%")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    print("Starting evaluation")
    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images, targets = batch
        images = images.to(device).contiguous().reshape(-1, *images.shape[-3:])
        targets = targets.to(device).contiguous().reshape(-1)
        print(f"Batch {batch_idx + 1}: Images shape after reshape: {images.shape}, Targets shape after reshape: {targets.shape}")

        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(images)
            loss = criterion(output, targets)

        loss_value = loss.item()
        print(f"Batch {batch_idx + 1} Evaluation Loss: {loss_value}")

        acc1, = accuracy(output, targets, topk=(1,))
        batch_size = images.size(0)
        metric_logger.update(loss=loss_value)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        print(f"Batch {batch_idx + 1} Evaluation Accuracy: {acc1.item() * 100:.2f}%")

    metric_logger.synchronize_between_processes()
    print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
