'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
'''
import argparse
import numpy as np
import random
import time

import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform


parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/demo/demo1.jpg')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')


if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=args.image_size)

    #######load model
    model = ram_plus(pretrained=args.pretrained,
                             image_size=args.image_size,
                             vit='swin_l')
    model.eval()

    model = model.to(device)

    start = time.time()
    image = transform(Image.open(args.image)).unsqueeze(0).to(device)
    res = inference(image, model)
    mid = time.time()
    print('图片1耗时: ', float(mid-start))
    print("Image Tags: ", res[0])
    print("图像标签: ", res[1])
    image2 = transform(Image.open('/mnt/chy/test/4.jpg')).unsqueeze(0).to(device)
    res2 = inference(image2, model)
    print('图片2耗时: ', float(time.time()-mid))
    print("Image Tags: ", res2[0])
    print("图像标签: ", res2[1])
