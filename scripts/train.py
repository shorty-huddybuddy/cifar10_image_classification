"""Train script for CIFAR-10 with multiple model choices.

Usage examples (PowerShell):
  python .\scripts\train.py --model resnet18 --epochs 10 --batch-size 128 --device cuda
  python .\scripts\train.py --model simple_cnn --epochs 5
"""
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from scripts.models import get_model
from scripts.utils import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in tqdm(loader, desc='train', leave=False):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='eval', leave=False):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += images.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple_cnn', help='model name: simple_cnn|resnet18|mobilenet_v2')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save-dir', type=str, default='./models')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, data_root=args.data_root)

    model = get_model(args.model, num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        # save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"{args.model}_epoch{epoch}.pth")
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'test_acc': test_acc}, ckpt_path)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'test_acc': test_acc}, os.path.join(args.save_dir, f"{args.model}_best.pth"))


if __name__ == '__main__':
    main()
