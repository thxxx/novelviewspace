import torch
import numpy as np
import json
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from diffusers import StableDiffusionPipeline
from utils import camera_position_to_line, sample_ray, inf_save
import tqdm
from accelerate.utils import set_seed
from torch import autocast
import random
import torch.optim as optim
from torch.utils.data import random_split
from sizespace import TextSpace, Text3DSpace
import argparse
import os

custom_position = [
    (-1.9, 1.8, 0.1),
    (-1.8, 1.7, 0.2),
    (-1.7, 1.5, 0.3),
    (-1.6, 1.4, 0.4),
    (-1.5, 1.3, 0.5),
    (-1.4, 1.2, 0.6),
    (-1.3, 1.1, 0.7),
    (-1.2, 0.9, 0.9),
    (-1.1, 0.8, 1.1),
    (-0.9, 0.7, 1.3),
    (-0.8, 0.6, 1.5),
    (-0.7, 0.5, 1.6),
    (-0.6, 0.4, 1.7),
    (-0.5, 0.3, 1.8),
    (-0.4, 0.1, 1.9),
    (-0.3, 0.1, 2),
    (-0.2, 0.1, 2),
    (-0.1, 0.1, 2),
    (-0.001, 0.1, 2),
    (0.1, 0.1, 2),
    (0.2, 0.1, 2),
    (0.3, 0.1, 1.9),
    (0.4, -0.1, 1.8),
    (0.5, -0.2, 1.7),
    (0.6, -0.3, 1.5),
    (0.7, -0.4, 1.4),
    (0.8, -0.5, 1.3),
    (0.9, -0.6, 1.2),
    (1.0, -0.7, 1.1),
    (1.1, -0.9, 0.9),
    (1.2, -1.1, 0.8),
    (1.3, -1.3, 0.7),
    (1.4, -1.5, 0.6),
    (1.5, -1.6, 0.5),
    (1.6, -1.7, 0.4),
    (1.7, -1.8, 0.3),
    (1.8, -1.9, 0.1),
]

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--layers_num", default=2, type=int, help=("Size of textual space"))
    parser.add_argument("--n_hiper", type=int, default=30, help=("What is the base embedding?"))
    parser.add_argument("--N_samples", default=12, type=int, help=("Learning rate for optimizer"))
    parser.add_argument("--output_dir", default="result9", type=str, help=("initial embedding for spaces."))
    parser.add_argument("--data_file", default="./ray.json", type=str, help=("initial embedding for spaces."))
    parser.add_argument("--custom", default=True, type=bool, help=("do inference for valid or custom position"))
    parser.add_argument("--model", default="model.pt", type=str, help=("do inference for valid or custom position"))
    parser.add_argument("--seed", default=1352, type=int, help=("do inference for valid or custom position"))

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    layers_num = args.layers_num
    n_hiper = args.n_hiper
    output_dir = args.output_dir
    N_samples = args.N_samples

    resolution=512

    model = Text3DSpace(layers_num, n_hiper).to('cuda')
    model_params = torch.load(f"{output_dir}/{args.model}")
    model.load_state_dict(model_params)

    on_line_positions=[]

    with open(args.data_file, "r") as js:
        rays = json.load(js)

    # if not args.custom:
    on_line_positions=camera_position_to_line(rays[:2], layers_num, "chairs", resolution)

    t_vals = torch.linspace(0., 1., steps=N_samples)
    near, far = 0, 2
    # near와 far 사이에서 N_samples 만큼 구역이 나뉨..!! 여기서 하나당 값은 샘플링 한게 아니라 그냥 구역 나눈거.
    z_vals = near * (1.-t_vals) + far * (t_vals)
    # z_vals = z_vals.expand([N_rays, N_samples]) # ray갯수(배치 사이즈) 만큼 복사

    # get intervals between samples
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1]) # 각 구역의 중간값. 2.0317, 2.00952, 2.1587 ... [N_samples - 1 개]
    upper = torch.cat([mids, z_vals[...,-1:]], -1) # 각 구역의 중간값, 6.000이 추가됨 2.0317, 2.00952, 2.1587 ... 6.000 [N_samples개]
    lower = torch.cat([z_vals[...,:1], mids], -1) # 각 구역의 시작값 2.000, 2.0317, 2.0952 ... [N_samples개]

    print("N_samples개만큼 샘플링 ,", z_vals)

    # if not args.custom:
    class CameraDataset(torch.utils.data.Dataset):
        def __init__(self):
            super(CameraDataset, self).__init__()
            self.cameras=on_line_positions
        
        def __len__(self):
            return len(on_line_positions)
        
        def __getitem__(self, idx):
            return on_line_positions[idx]

    dataset = CameraDataset()
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.9)
    validation_size = dataset_size - train_size
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, 0])
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=True)
    print("dataset : ", len(dataset))

    def train(model):
        epochs=1001
        adam_beta1=0.9
        adam_beta2=0.999
        adam_epsilon=1e-08
        emb_learning_rate=0.001
        emb_train_steps=1
        pretrained_model_name = "CompVis/stable-diffusion-v1-4"
        # mixed_precision="fp16"
        mixed_precision="no"
        gradient_accumulation_steps=1
        # Setting
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
        weight_dtype = torch.float32
        if mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        set_seed(args.seed)
        g_cuda = torch.Generator(device='cuda')
        g_cuda.manual_seed(args.seed)
        # ---------------settings--------------- #

        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name, subfolder="tokenizer", use_auth_token=True)
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder="unet", use_auth_token=True)
        # noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        # Encode the input image.
        vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae", use_auth_token=True)
        CLIP_text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name, subfolder="text_encoder", use_auth_token=True)
        vae.to(accelerator.device, dtype=weight_dtype)
        
        # Encode the source and target text.
        CLIP_text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_ids_src = tokenizer("mic", padding="max_length",truncation=True,max_length=tokenizer.model_max_length,return_tensors="pt").input_ids
        text_ids_src = text_ids_src.to(device=accelerator.device)
        with torch.inference_mode():
            source_embeddings = CLIP_text_encoder(text_ids_src)[0].float()
        # del vae, CLIP_text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # For inference
        ddim_scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name, scheduler=ddim_scheduler, torch_dtype=torch.float16).to("cuda")
        num_samples = 2
        guidance_scale = 7.5 
        num_inference_steps = 50
        height = 512
        width = 512
        optimizer_class = torch.optim.Adam
        src_embeddings = source_embeddings[:,:-n_hiper].clone().detach()

        model.to(accelerator.device, dtype=weight_dtype)
        optimizer = optimizer_class(
            model.parameters(),
            lr=emb_learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
        )
        unet, optimizer = accelerator.prepare(unet, optimizer)

        inf_emb = []
        inf_images=[]    
        # for idx, data in enumerate(dataloader):
        #     now_position = data['camera_position_on_line']
        #     last_position = torch.tensor([-now_position[0], -now_position[1]])
        #     sample_rays = sample_ray(z_vals, now_position, upper, lower, inf=True) # N_samples개가 나옴 inference 때는 균등하게
        #     sample_rays.to(accelerator.device)
        #     new_view = model(sample_rays, last_position) # get interpolated one by 4 samples
        #     new_view_embeddings = new_view.unsqueeze(dim=0).to(accelerator.device, dtype=weight_dtype)

        #     input_image = data['input_image'].squeeze()

        #     inf_emb = []
        #     inf_emb.append(torch.cat([src_embeddings, new_view_embeddings.clone().detach()], 1))
            
        #     inf_images.append(input_image)
        #     with autocast("cuda"), torch.inference_mode():
        #         seed=random.randrange(1,10000)
        #         set_seed(seed)
        #         g_cuda = torch.Generator(device='cuda')
        #         g_cuda.manual_seed(seed)
        #         for embs in inf_emb:
        #             images = pipe(prompt_embeds=embs, height=height, width=width, num_images_per_prompt=num_samples,
        #                 num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=g_cuda)
        #             for j in range(len(images.images)):
        #                 if not images.nsfw_content_detected[j]:
        #                     inf_images.append(images.images[j])
        #                     break
            
        #     save_path=os.path.join(f"{output_dir}/novel2/epoh{idx}_image.png")
        #     if idx%3==2:
        #         inf_save(inf_images, ["test" for u in range(len(inf_images))], save_path)
        #         inf_images=[]


        for now_position in custom_position:
            print("이미지 생성, 위치는 : ", now_position)
            new_last_position = torch.tensor([-now_position[0], -now_position[1], -now_position[2]])
            new_sample_rays = sample_ray(z_vals, now_position, upper, lower, inf=True) # N_samples개가 나옴 inference 때는 균등하게
            new_sample_rays.to(accelerator.device)
            new_view = model(new_sample_rays, new_last_position) # get interpolated one by 4 samples
            new_view_embeddings = new_view.unsqueeze(dim=0).to(accelerator.device, dtype=weight_dtype)
            
            inf_emb.append(torch.cat([src_embeddings, new_view_embeddings.clone().detach()], 1))

        def make_images_by_embed(inf_emb):
            inf_images=[]
            with autocast("cuda"), torch.inference_mode():
                seed=random.randrange(1,10000)
                set_seed(seed)
                g_cuda = torch.Generator(device='cuda')
                g_cuda.manual_seed(seed)
                for embs in inf_emb:
                    images = pipe(prompt_embeds=embs, height=height, width=width, num_images_per_prompt=num_samples,
                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=g_cuda)
                    for j in range(len(images.images)):
                        if not images.nsfw_content_detected[j]:
                            inf_images.append(images.images[j])
                            break
            for idx, image in enumerate(inf_images):
                image.save(f"{output_dir}/novel3/nover_view_{idx}.png")

        make_images_by_embed(inf_emb) 

    train(model)

if __name__ == "__main__":
    main()