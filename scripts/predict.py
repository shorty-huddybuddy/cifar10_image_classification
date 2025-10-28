"""Predict script: run per-image predictions using a saved checkpoint.

Usage (PowerShell):
  python .\scripts\predict.py --checkpoint ./models/resnet18_best.pth --input-folder ./images --output ./predictions.csv --device cpu
"""
import argparse
from pathlib import Path
import csv
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from scripts.models import get_model
from scripts.utils import CIFAR10_CLASSES


def load_image(path, transform):
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output', type=str, default='predictions.csv')
    parser.add_argument('--model', type=str, default='resnet18', help='model architecture used for checkpoint')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    # transform: resize to 32 and use CIFAR normalization
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    model = get_model(args.model, num_classes=len(CIFAR10_CLASSES))
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.to(device)
    model.eval()

    input_folder = Path(args.input_folder)
    image_paths = sorted([p for p in input_folder.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
    if not image_paths:
        print('No images found in', input_folder)
        return

    results = []
    with torch.no_grad():
        for p in image_paths:
            x = load_image(p, transform).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().squeeze(0)
            top1 = probs.argmax().item()
            results.append((str(p.name), CIFAR10_CLASSES[top1], float(probs[top1].item())))

    # write CSV
    out_path = Path(args.output)
    with out_path.open('w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'predicted_class', 'probability'])
        for row in results:
            writer.writerow(row)

    print(f'Wrote predictions for {len(results)} images to {out_path}')


if __name__ == '__main__':
    main()
