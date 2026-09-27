"""Train or evaluate a ResNet-18 beauty regressor on SCUT-FBP5500.

You must obtain the dataset yourself and accept its terms (non-commercial research
use only; see NOTICE.md). Expected layout, as in the dataset's own split files:

    <data-root>/<image file>          e.g. faces/AF1.jpg
    <list file>                       one "<image file> <score>" per line

Recipe (from the SCUT-FBP5500 paper): ImageNet-pretrained ResNet-18, resize 256 then
random-crop 224, L2 loss, SGD lr 0.001 / momentum 0.9 / weight decay 1e-4, batch 16,
lr /10 every 5000 iterations, 20000 iterations.

    python scripts/train.py --data-root data/faces --train-list data/1/train_1.txt \
        --test-list data/1/test_1.txt --out weights/resnet18.pth
    python scripts/train.py --data-root data/faces --test-list data/1/test_1.txt \
        --eval-only --weights weights/resnet18.pth
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import IMAGENET_MEAN, IMAGENET_STD, load_weights  # noqa: E402

NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
TRAIN_TF = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), NORMALIZE])
EVAL_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), NORMALIZE])


class ListDataset(Dataset):
    def __init__(self, root: str, list_file: str, transform) -> None:
        with open(list_file, encoding="utf-8") as handle:
            rows = [line.split() for line in handle if line.strip()]
        self.items = [(os.path.join(root, name), float(score)) for name, score in rows]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, score = self.items[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor([score], dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    predictions, labels = [], []
    for images, targets in loader:
        predictions.append(model(images.to(device)).cpu().squeeze(1))
        labels.append(targets.squeeze(1))
    pred, label = torch.cat(predictions).numpy(), torch.cat(labels).numpy()
    return {
        "pearson": float(np.corrcoef(pred, label)[0, 1]),
        "mae": float(np.mean(np.abs(pred - label))),
        "rmse": float(np.sqrt(np.mean((pred - label) ** 2))),
    }


def train(args: argparse.Namespace, device: str) -> None:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    loader = DataLoader(ListDataset(args.data_root, args.train_list, TRAIN_TF), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=args.lr_step, gamma=0.1)
    criterion = nn.MSELoss()

    iteration = 0
    model.train()
    while iteration < args.iterations:
        for images, targets in loader:
            loss = criterion(model(images.to(device)), targets.to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            iteration += 1
            if iteration % 100 == 0:
                print(f"iter {iteration}/{args.iterations} loss {loss.item():.4f}")
            if iteration >= args.iterations:
                break

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.out)
    print(f"saved {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-list")
    parser.add_argument("--test-list")
    parser.add_argument("--out", default="weights/resnet18.pth")
    parser.add_argument("--weights", help="checkpoint to evaluate (with --eval-only)")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--lr-step", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.eval_only:
        if not args.train_list:
            parser.error("--train-list is required unless --eval-only")
        train(args, device)
        args.weights = args.out

    if args.test_list:
        model = load_weights(args.weights or args.out, device)
        loader = DataLoader(ListDataset(args.data_root, args.test_list, EVAL_TF), batch_size=32, num_workers=args.workers)
        metrics = evaluate(model, loader, device)
        print("Pearson {pearson:.4f}  MAE {mae:.4f}  RMSE {rmse:.4f}".format(**metrics))


if __name__ == "__main__":
    main()
