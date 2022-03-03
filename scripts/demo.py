import os
import sys
import argparse

import cv2
import numpy as np
import torch

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from core.utils.visualize import get_color_pallete
from core.models import get_model

# parser = argparse.ArgumentParser(
#     description='Predict segmentation result from a given image')
# parser.add_argument('--model', type=str, default='psp_resnet50_voc',
#                     help='model name (default: fcn32_vgg16)')
# parser.add_argument('--dataset', type=str, default='pascal_aug', choices=['pascal_voc, pascal_aug, ade20k, citys'],
#                     help='dataset name (default: pascal_voc)')
# parser.add_argument('--save-folder', default='../models',
#                     help='Directory for saving checkpoint models')
# parser.add_argument('--input-pic', type=str, default='../image/2007_002597.jpg',
#                     help='path to the input picture')
# parser.add_argument('--outdir', default='./eval', type=str,
#                     help='path to save the predict result')
# parser.add_argument('--local_rank', type=int, default=0)
# args = parser.parse_args()
#
#
# def demo(config):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # output folder
#     if not os.path.exists(config.outdir):
#         os.makedirs(config.outdir)
#
#     # image transform
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])
#     image = Image.open(config.input_pic).convert('RGB')
#     images = transform(image).unsqueeze(0).to(device)
#
#     model = get_model(args.model, pretrained=True, root=args.save_folder).to(device)
#     print('Finished loading model!')
#
#     model.eval()
#     with torch.no_grad():
#         output = model(images)
#
#     pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
#     mask = get_color_pallete(pred, args.dataset)
#     outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '.png'
#     mask.save(os.path.join(args.outdir, outname))
#
#
# if __name__ == '__main__':
#     demo(args)
#
#
def demo(model, dataset, input_pic, outdir='./eval', save_folder='../models', local_rank=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(input_pic).convert('RGB')
    images = transform(image).unsqueeze(0).to(device)
    print(model)
    model = get_model(model, pretrained=True, root=save_folder).to(device)
    print('Finished loading model!')

    model.eval()
    with torch.no_grad():
        output = model(images)

    pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred, dataset)
    outname = os.path.splitext(os.path.split(input_pic)[-1])[0] + '.png'
    mask.save(os.path.join(outdir, outname))
    img = cv2.imread(os.path.join(outdir, outname))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(1)
    # img2 = np.zeros_like(img)
    # img2[:, :, 0] = gray
    # img2[:, :, 1] = gray
    # img2[:, :, 2] = gray
    # mask = img2
    # mask = Image.open(os.path.join(outdir, outname))
    # a = mask.size[0]
    # b = mask.size[1]
    # im1 = im2 = im3 = mask.convert('L')
    # mask = Image.merge('RGB', (im1, im2, im3))
    # mask = np.asarray(mask).reshape(a, b, 3)
    return img
#
if __name__ == '__main__':
    demo('fcn8s_vgg16_voc', 'pascal_voc', '../image/2007_002597.jpg')