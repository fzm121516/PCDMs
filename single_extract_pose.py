from src.controlnet_aux import DWposeDetector
from PIL import Image
import torchvision.transforms as transforms
import torch


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


# def inference_pose(img_path, image_size=(1024, 1024), save_path='./data/output_img.png'):
#     device = torch.device("cuda:0")  # 确保这里的设备索引正确
#     model = init_dwpose_detector(device=device)
#     pil_image = Image.open(img_path).convert("RGB").resize(image_size, Image.BICUBIC)
#     dwpose_image = model(pil_image, output_type='np', image_resolution=image_size[1])
#     save_dwpose_image = Image.fromarray(dwpose_image)
#     save_dwpose_image.save(save_path)
#     print(f"Saved inference result to {save_path}")

def inference_pose(img_path, save_path='./data/output_img.png'):
    # 指定使用的设备为CUDA（GPU），索引为0
    device = torch.device("cuda:0")  # 确保这里的设备索引正确

    # 初始化 dwpose 检测器模型，传递设备参数
    model = init_dwpose_detector(device=device)

    # 打开图像文件并转换为RGB模式
    pil_image = Image.open(img_path).convert("RGB")

    # 获取输入图像的尺寸
    image_width, image_height = pil_image.size

    # 将图像传入模型进行姿态推理，指定输出类型为'numpy'数组，且图像分辨率为输入图像的宽度
    dwpose_image = model(pil_image, output_type='np', image_resolution=image_width)

    # 将 numpy 数组转换回 PIL 图像
    save_dwpose_image = Image.fromarray(dwpose_image)

    # 保存生成的图像到指定路径
    save_dwpose_image.save(save_path)

    # 打印保存成功的信息
    print(f"Saved inference result to {save_path}")


# 指定输入图像的路径
path = './data/img1.png'

# 指定保存推理结果图像的路径
save_path = './data/output_img1.png'

# 调用推理函数，传入图像路径和保存路径
inference_pose(path, save_path=save_path)

path = './data/img1.png'
save_path = './data/output_img1.png'

inference_pose(path, save_path=save_path)
