import argparse
import torch
from pathlib import Path
from PIL import Image
from utils import get_transform
from models import CustomNet, resnet18, vgg16
from constants import CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', type=str,
                        default=None, help='Path to an input file.')
    parser.add_argument('--model', '-m', type=str, choices=['custom', 'vgg16', 'resnet18'],
                        default='resnet18', help='Model name.')
    args = parser.parse_args()
    input_file_name = args.input_file
    model_name = args.model
    img = Image.open(input_file_name)
    if model_name == 'resnet18Fer2013':
        net = resnet18()
        weights = 'resnet18Fer2013.pth'
        transform = get_transform(48)
    elif model_name == 'resnet18KDEF':
        net = resnet18()
        weights = 'resnet18KDEF.pth'
        transform = get_transform(224)
    elif model_name == 'resnet18Dartmouth':
        net = resnet18()
        weights = 'resnet18Dartmouth.pth'
        transform = get_transform(224)
    path = Path.joinpath(Path(), 'weights', weights)
    net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    net.eval()
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        preds = softmax(net(img_tensor)).numpy().ravel()
        for i in range(len(CLASSES)):
            print('{:>8}: {:5.2f}%'.format(CLASSES[i], 100*preds[i]))


if __name__ == '__main__':
    main()
