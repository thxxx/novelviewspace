import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import lpips
import math
import torchvision

#codes for 'save_image' and 'text_under_image' are from 
# https://github.com/google/prompt-to-prompt/blob/main/prompt-to-prompt_stable.ipynb

def save_images(images, num_rows=1, offset_ratio=0.02, name=None):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    pil_img.save(name)
    
    
def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def inf_save(inf_img, name, save_name):
    images = []
    for i in range(len(inf_img)):
        image = np.array(inf_img[i].resize((256,256)))
        image = text_under_image(image, name[i])
        images.append(image)
    save_images(np.stack(images, axis=0), name = save_name)


def process_img(image_root, resolution):
    input_image = Image.open(image_root)
    input_image=np.array(input_image)
    # print(f"\n\n {image_root} , {input_image.shape} \n\n")
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            if input_image[i][j][-1] == 0:
                input_image[i][j]=[127,127,127,255]
    input_image=Image.fromarray(input_image)
    input_image=input_image.convert("RGB")
    image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if True else transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image_to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    init_image = image_transforms(input_image)
    input_image = image_to_tensor(input_image)
    return init_image, input_image

def camera_position_to_line(rays, layers_num, data_dir, resolution=512):
    """
    return img_path, position
    """
    on_line_positions=[]
    for ray in rays:
        x, y, z = ray['coord']['x'], ray['coord']['y'], ray['coord']['z']
        # z의 평면에 projected point가 존재할 조건 : z가 x, y보다 크면 됨.
        if abs(z)>abs(x) and abs(z)>abs(y):
            x = layers_num * x / abs(z)
            y = layers_num * y / abs(z)
            if z<0:
                z = -layers_num
            else:
                z = layers_num
        else : 
            if abs(x)>abs(y):
                z = z * layers_num / abs(x)
            elif abs(y)>=abs(x):
                z = z * layers_num / abs(y)
            if x>0 and y>0: # 1사
                if x>y:
                    x, y = layers_num, (y/x)*layers_num
                elif y>=x:
                    x, y = (x/y)*layers_num, layers_num
            if x<0 and y<0: # 3사 
                if x>y: #abs(x)<abs(y)
                    x, y = -(x/y)*layers_num, -layers_num
                elif y>=x: # abs(x)>=abs(y)
                    x, y = -layers_num, -(y/x)*layers_num
            if x>0 and y<0: # 4사
                if abs(x)>abs(y):
                    x, y = layers_num, (y/x)*layers_num
                elif abs(y)>=abs(x):
                    x, y = (x/y)*layers_num, -layers_num
            if x<0 and y>0: # 2사
                if abs(x)>abs(y):
                    x, y = -layers_num, -(y/x)*layers_num
                elif abs(y)>=abs(x):
                    x, y = (x/y)*layers_num, layers_num

        image_root=f"./dataset/{data_dir}/{ray['img_path']}"

        device="cuda"
        weight_dtype = torch.float32
        init_image, input_image = process_img(image_root, resolution)

        on_line_positions.append({ 
            "img_path": ray['img_path'], 
            "origin_position": (ray['coord']['x'], ray['coord']['y'], ray['coord']['z']),
            "camera_position_on_line" : (x, y, z),
            "init_image": init_image,
            "input_image": input_image,
        })
    return on_line_positions

def sample_ray(z_vals, camera_position_on_line, upper, lower, inf=False):
    # stratified samples in those intervals
    t_rand = torch.rand(z_vals.shape) if not inf else torch.tensor([0.5]*z_vals.shape[0]) # [N_samples]개의 랜덤값을 얻음 그냥.
    z_vals2 = lower + (upper - lower) * t_rand # 2.000 + 범위(0.0317) * 랜덤값(0~1) -> 2.000과 2.0317 사이에 랜덤값이 선택됨
    x, y, z = camera_position_on_line
    sampled_positions = []
    for zval in z_vals2:
        sampled_x = x - x*zval.item()
        sampled_y = y - y*zval.item()
        sampled_z = z - z*zval.item()
        sampled_positions.append((sampled_x, sampled_y, sampled_z))
    
    # plt.scatter(x, y, c="r")
    # plt.scatter(-x, -y, c="b")
    # plt.scatter([a[0] for a in sampled_positions], [a[1] for a in sampled_positions], c="g")
    # plt.title(f'sampled')
    # plt.xlabel('x-axis')
    # plt.ylabel('y-axis')
    # plt.xlim(-2.1, 2.1)
    # plt.ylim(-2.1, 2.1)
    # plt.savefig(f'samples{x+y}.png')
    # plt.cla()
    return torch.tensor(sampled_positions)

def get_metrics(img1, img2, loss_fn):
    img1=img1.cpu().detach().squeeze()
    img2=img2*2-1
    # lpips
    # print(f"\ntype : {img1.shape}, {img2.shape}\n") 
    if not loss_fn:
        loss_fn = lpips.LPIPS(net='alex')
    lpip_s = loss_fn.forward(img1, img2)

    # PSNR
    msed=0
    count=1
    msed2=0
    # for j in range(img1.shape[1]):
    #     for k in range(img1.shape[2]):
    #         value = abs(img1[0][j][k]-img2[0][j][k])**2 + abs(img1[1][j][k]-img2[1][j][k])**2 + abs(img1[2][j][k]-img2[2][j][k])**2
    #         msed += value
    #         if img1[2][j][k].item() == -0.003921568393707275 and img1[0][j][k].item() == -0.003921568393707275 and img1[1][j][k].item() == -0.003921568393707275:
    #             continue
    #         count+=1
    #         msed2 += value

    # print(f"\nmax(img1) : {torch.max(img1)}, max(img1) : {torch.max(img2)}")
    # print(f"min(img1) : {torch.min(img1)}, min(img1) : {torch.min(img2)}")

    # PSNR = 20*math.log10(2./math.sqrt(msed2/(3*count)))
    PSNR_nogray = 1

    MSE = F.mse_loss(img1, img2, reduction="mean")
    PSNR = 20*math.log10(2./math.sqrt(MSE+0.000001))

    return lpip_s.squeeze().item(), PSNR, PSNR_nogray


def just_line(rays, layers_num, data_dir, resolution=512):
    """
    return img_path, position
    """
    on_line_positions=[]
    for ray in rays:
        x, y, z = ray['coord']['x'], ray['coord']['y'], ray['coord']['z']
        # z의 평면에 projected point가 존재할 조건 : z가 x, y보다 크면 됨.
        print("zs : ", z)
        if abs(z)>abs(x) and abs(z)>abs(y):
            x = layers_num * x / abs(z)
            y = layers_num * y / abs(z)
            if z<0:
                print("나오면 안됨")
                z = -layers_num
            else:
                z = layers_num
        else:
            if abs(x)>abs(y):
                z = z * layers_num / abs(x)
            elif abs(y)>=abs(x):
                z = z * layers_num / abs(y)
            if x>0 and y>0: # 1사
                if x>y:
                    x, y = layers_num, (y/x)*layers_num
                elif y>=x:
                    x, y = (x/y)*layers_num, layers_num
            if x<0 and y<0: # 3사 
                if x>y: #abs(x)<abs(y)
                    x, y = -(x/y)*layers_num, -layers_num
                elif y>=x: # abs(x)>=abs(y)
                    x, y = -layers_num, -(y/x)*layers_num
            if x>0 and y<0: # 4사
                if abs(x)>abs(y):
                    x, y = layers_num, (y/x)*layers_num
                elif abs(y)>=abs(x):
                    x, y = (x/y)*layers_num, -layers_num
            if x<0 and y>0: # 2사
                if abs(x)>abs(y):
                    x, y = -layers_num, -(y/x)*layers_num
                elif abs(y)>=abs(x):
                    x, y = (x/y)*layers_num, layers_num

        image_root=f"./dataset/{data_dir}/{ray['img_path']}"

        device="cuda"
        weight_dtype = torch.float32

        on_line_positions.append({ 
            "img_path": ray['img_path'], 
            "origin_position": (ray['coord']['x'], ray['coord']['y'], ray['coord']['z']),
            "camera_position_on_line" : (x, y, z),
        })
    return on_line_positions