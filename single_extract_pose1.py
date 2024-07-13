from src.controlnet_aux import DWposeDetector
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

import argparse
import os
import glob

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


def init_dwpose_detector(device):
    # specify configs, ckpts and device, or it will be downloaded automatically and use cpu by default
    det_config = './src/configs/yolox_l_8xb8-300e_coco.py'
    det_ckpt = './ckpts/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
    pose_config = './src/configs/dwpose-l_384x288.py'
    pose_ckpt = './ckpts/dw-ll_ucoco_384.pth'

    dwpose_model = DWposeDetector(
        det_config=det_config,
        det_ckpt=det_ckpt,
        pose_config=pose_config,
        pose_ckpt=pose_ckpt,
        device=device
    )
    return dwpose_model.to(device)


# def inference_pose(img_path, image_size=(1024, 1024)):
#     device = torch.device(f"cuda:{0}")
#     model = init_dwpose_detector(device=device)
#     pil_image = Image.open(img_path).convert("RGB").resize(image_size, Image.BICUBIC)
#     dwpose_image = model(pil_image, output_type='np', image_resolution=image_size[1])
#     save_dwpose_image = Image.fromarray(dwpose_image)
#     return save_dwpose_image


device = torch.device(f"cuda:{0}")
model = init_dwpose_detector(device=device)
image_size = (1024, 1024)




# # 指定输入图像的路径
# path = './data/img1.png'
#
# # 指定保存推理结果图像的路径
# save_path = './data/output_img1.png'
#
# # 调用推理函数，传入图像路径和保存路径
# inference_pose(path).save(save_path)

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--result-dir', type=str, required=True)


args = parser.parse_args()



# Load Images
image_list = sorted([*glob.glob(os.path.join(args.images_dir, '**', '*.jpg'), recursive=True),
                     *glob.glob(os.path.join(args.images_dir, '**', '*.png'), recursive=True)])



num_image = len(image_list)
print("Find ", num_image, " images")

# Process
for i in range(num_image):
    image_path = image_list[i]
    image_name = image_path[image_path.rfind('/') + 1:image_path.rfind('.')]
    print(i, '/', num_image, image_name)



    pil_image = Image.open(image_path).convert("RGB").resize(image_size, Image.BICUBIC)
    dwpose_image = model(pil_image, output_type='np', image_resolution=image_size[1])
    save_dwpose_image = Image.fromarray(dwpose_image)

    # inference
    pred_alpha = save_dwpose_image

    # save results
    output_dir = args.result_dir + image_path[len(args.images_dir):image_path.rfind('/')]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 拼接保存路径并创建
    save_path = os.path.join(output_dir, image_name + '.png')

    # 确保保存路径存在并保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pred_alpha.save(save_path)
    # Image.fromarray(((pred_alpha * 255).astype('uint8')), mode='L').save(save_path)
