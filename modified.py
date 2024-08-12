import torch
import json
import numpy as np
import math
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from diffusers import StableDiffusionPipeline
from utils import process_img
from utils import inf_save, camera_position_to_line, sample_ray, get_metrics
# from tqdm.auto import tqdm
import tqdm
from accelerate.utils import set_seed
import torch.nn.functional as F
from torch import autocast
import random
import os
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import random_split
from sizespace import TextSpace, Text3DSpace, Text3DSpaceAll
import argparse
import torchvision
from datetime import datetime
import lpips

loss_fn = lpips.LPIPS(net='alex')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--layers_num", default=2, type=int,
                        help=("Size of textual space"))
    parser.add_argument("--n_hiper", type=int, default=30,
                        help=("What is the base embedding?"))
    parser.add_argument("--epochs", type=int, default=1601,
                        help=("How many epochs to iterate"))
    parser.add_argument("--learning_rate", default=0.003,
                        type=float, help=("Learning rate for optimizer"))
    parser.add_argument("--N_samples", default=12, type=int,
                        help=("Learning rate for optimizer"))
    parser.add_argument("--initial_embedding", default="",
                        type=str, help=("initial embedding for spaces."))
    parser.add_argument("--output_dir", default="./results/result9", type=str, help=("initial embedding for spaces."))
    # parser.add_argument("--dataset", default="./dataset/ray.json", type=str, help=("dataset location."))
    parser.add_argument("--seed", default=1323, type=int, help=("seed for sdm"))
    parser.add_argument("--object", default="hotdog", type=str, help=("what is the object?"))
    parser.add_argument("--continue_model", default="", type=str, help=("start from what model?"))
    parser.add_argument("--start_epoch", default=0, type=int, help=("start epoch"))
    parser.add_argument("--train_set", default="trains_hotdog.json", type=str, help=("what is the object?"))
    parser.add_argument("--valid_set", default="valids_hotdog.json", type=str, help=("what is the object?"))

    args = parser.parse_args()
    return args

def main():
    # 1. 카메라의 위치를 input으로 받는다. # 어짜피 중앙(원점)을 지나는 ray로 샘플링 하는데 위치말고 카메라가 바라보는 방향을 받아야하는 이유가 있나? 잘 모르겠다.
    # 2. 해당 위치에서 중앙을 지나는 ray를 긋고, n_samples등분 한다.
    # 3. 각 구간의 random 포지션을 선택하고 해당 위치에서 interpolation을 계산하여 [30, 768]을 구한다.
    # 4. 그렇게 얻은 n_samples를 합쳐서 하나의 embedding[30, 768]으로 만든다.
    # 5. 이 embedding으로 diffusion한다.
    args = parse_args()

    layers_num = args.layers_num
    n_hiper = args.n_hiper
    output_dir = args.output_dir
    N_samples = args.N_samples

    resolution = 512

    model = Text3DSpaceAll(layers_num=layers_num, n_hiper=n_hiper)

    if len(args.continue_model) > 3:
        print(f"load trained model")
        model.load_state_dict(torch.load(args.continue_model))

    with open(f"{args.train_set}", "r") as js:
        train_rays = json.load(js)
        train_rays = [{
            "img_path":t['img_path'],
            "coord":{
                "x":t['origin_position'][0],
                "y":t['origin_position'][1],
                "z":t['origin_position'][2],
            }
        } for t in train_rays[:59]]
        print("train ray dataset length : ", len(train_rays))
    
    with open(f"{args.valid_set}", "r") as js:
        valid_rays = json.load(js)
        valid_rays = [{
            "img_path":v['img_path'],
            "coord":{
                "x":v['origin_position'][0],
                "y":v['origin_position'][1],
                "z":v['origin_position'][2],
            }
        } for v in valid_rays]
        print("valid ray dataset length : ", len(valid_rays))

    on_line_positions_train = camera_position_to_line(train_rays, layers_num, "hotdogs", resolution)
    on_line_positions_valid = camera_position_to_line(valid_rays, layers_num, "hotdogs", resolution)

    t_vals = torch.linspace(0., 1., steps=N_samples)
    near, far = 0, 2
    # near와 far 사이에서 N_samples 만큼 구역이 나뉨..!! 여기서 하나당 값은 샘플링 한게 아니라 그냥 구역 나눈거.
    z_vals = near * (1.-t_vals) + far * (t_vals)

    # get intervals between samples
    # 각 구역의 중간값. 2.0317, 2.00952, 2.1587 ... [N_samples - 1 개]
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    # 각 구역의 중간값, 6.000이 추가됨 2.0317, 2.00952, 2.1587 ... 6.000 [N_samples개]
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    # 각 구역의 시작값 2.000, 2.0317, 2.0952 ... [N_samples개]
    lower = torch.cat([z_vals[..., :1], mids], -1)

    def train(model):
        adam_beta1 = 0.9
        adam_beta2 = 0.999
        adam_epsilon = 1e-08
        emb_train_steps = 1
        pretrained_model_name = "CompVis/stable-diffusion-v1-4"
        # mixed_precision="fp16"
        mixed_precision = "no"
        gradient_accumulation_steps = 1
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
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name, subfolder="tokenizer", use_auth_token=True)
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name, subfolder="unet", use_auth_token=True)
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        # Encode the input image.
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name, subfolder="vae", use_auth_token=True)
        CLIP_text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name, subfolder="text_encoder", use_auth_token=True)
        vae.to(accelerator.device, dtype=weight_dtype)

        # Encode the source and target text.
        CLIP_text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_ids_src = tokenizer(args.object, padding="max_length", truncation=True,
                                 max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
        text_ids_src = text_ids_src.to(device=accelerator.device)
        with torch.inference_mode():
            source_embeddings = CLIP_text_encoder(text_ids_src)[0].float()
        # del vae, CLIP_text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # For inference
        ddim_scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                       beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name, scheduler=ddim_scheduler, torch_dtype=torch.float16, safety_checker = None).to(accelerator.device)
        num_samples = 3
        guidance_scale = 7.5
        num_inference_steps = 50
        height = 512
        width = 512
        optimizer_class = torch.optim.Adam
        src_embeddings = source_embeddings[:, :-n_hiper].clone().detach()

        model.to(accelerator.device, dtype=weight_dtype)
        optimizer = optimizer_class(
            model.parameters(),
            lr=args.learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        unet, optimizer = accelerator.prepare(unet, optimizer)

        class CameraDataset(torch.utils.data.Dataset):
            def __init__(self, base_sets):
                super(CameraDataset, self).__init__()
                new_data = []
                for d in base_sets:
                    init_image = d['init_image'].squeeze()
                    init_image = init_image[None].to(
                        device=accelerator.device, dtype=weight_dtype)
                    with torch.inference_mode():
                        init_latents = vae.encode(
                            init_image).latent_dist.sample()
                        init_latents = 0.18215 * init_latents
                    dicto = d
                    dicto['init_latents'] = init_latents
                    new_data.append(dicto)

                self.cameras = new_data

            def __len__(self):
                return len(self.cameras)

            def __getitem__(self, idx):
                return self.cameras[idx]

        train_dataset = CameraDataset(on_line_positions_train)
        validation_dataset = CameraDataset(on_line_positions_valid)

        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

        print("dataset length : ", len(train_dataset))

        def make_image(inf_emb):
            with autocast("cuda"), torch.inference_mode():
                seed = random.randrange(1, 10000)
                set_seed(seed)
                g_cuda = torch.Generator(device='cuda')
                g_cuda.manual_seed(seed)
                images = pipe(prompt_embeds=inf_emb, height=height, width=width, num_images_per_prompt=num_samples,
                                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=g_cuda)
                for j in range(len(images.images)):
                    return images.images[j]

        def wrtie_to_history(text):
            with open(f"{output_dir}/history.txt", "a") as file:
                file.write(text)

        temp_loss = []
        for epoch in range(args.start_epoch, args.epochs):
            for i, samples in enumerate(tqdm.tqdm(dataloader)):
                # for 3d
                last_position = torch.tensor([-s for s in samples['camera_position_on_line']])
                sample_rays = sample_ray(z_vals,  samples['camera_position_on_line'], upper, lower)  # N_samples개의 임베딩이 나옴
                sample_rays.to(accelerator.device)
                # Load image and set transparent background to gray
                init_image, input_image = samples['init_image'].squeeze(), samples['input_image'].squeeze()
                init_latents = samples['init_latents'].squeeze(0)
                
                with accelerator.accumulate(unet):
                    noise = torch.randn_like(init_latents)
                    bsz = init_latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=init_latents.device)
                    timesteps = timesteps.long()

                    noisy_latents = noise_scheduler.add_noise(init_latents, noise, timesteps)

                    # 이게 들어가서 interpolation, merge -> 하나의 임베딩 리
                    hiper_embeddings = model(sample_rays, last_position)
                    hiper_embeddings = hiper_embeddings.unsqueeze(dim=0).to(accelerator.device, dtype=weight_dtype)

                    # src_embeddings는 optimize 안되는거 확인
                    source_embeddings = torch.cat([src_embeddings, hiper_embeddings], 1)
                    noise_pred = unet(noisy_latents, timesteps, source_embeddings).sample
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    temp_loss.append(loss.cpu().detach().item())

                    accelerator.backward(loss)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if epoch % 100 == 99 and i < 3:
                    print("100에퐄이라 그림을 그려 보겠습니다.")
                    input_image = to_pil_image(input_image)
                    inf_emb = torch.cat([src_embeddings, hiper_embeddings.clone().detach()], 1)

                    inf_images = [input_image]
                    made_image = make_image(inf_emb)
                    inf_images.append(made_image)
                    
                    to_tensor = torchvision.transforms.ToTensor()
                    lpips, PSNR, PSNR_nogray = get_metrics(init_image, to_tensor(made_image), loss_fn)
                    wrtie_to_history(f"\nepoch : {epoch} train image metric : lpips : {lpips}, PSNR : {PSNR}, PSNR_nogray : {PSNR_nogray}")

                    save_path = os.path.join(f"{output_dir}/epoch_{epoch}_th{i}_image.png")
                    inf_save(inf_images, [f'loss:{np.mean(temp_loss)}', f'[src, hper]'], save_path)

            if epoch % 20 == 19:
                wrtie_to_history(f"\n{ str(datetime.now())} , epoch :{epoch}")

            if epoch % 100 == 0:
                wrtie_to_history("\n\n" + f"epoch :{epoch}, temp_loss : {np.mean(temp_loss)}, lr : {optimizer.param_groups[0]['lr']}")
                inf_emb = []
                inf_images = []

                lpips_list = []
                PSNR_list = []
                PSNR_nogray_list = []

                for idx, valid_data in enumerate(valid_dataloader):
                    if idx > 7:
                        continue
                    new_last_position = torch.tensor([-v for v in valid_data['camera_position_on_line']])
                    new_sample_rays = sample_ray(z_vals, valid_data['camera_position_on_line'], upper, lower)  # N_samples개가 나옴
                    new_sample_rays.to(accelerator.device)
                    # get interpolated one by 12 samples
                    new_view = model(new_sample_rays, new_last_position)
                    new_view_embeddings = new_view.unsqueeze(dim=0).to(accelerator.device, dtype=weight_dtype)

                    embed = torch.cat([src_embeddings, new_view_embeddings.clone().detach()], 1)
                    inf_images.append(to_pil_image(valid_data['input_image'].squeeze()))
                    made_image = make_image(embed)
                    inf_images.append(made_image)

                    init_img = valid_data['init_image'].squeeze()
                    to_tensor = torchvision.transforms.ToTensor()
                    lpips, PSNR, PSNR_nogray = get_metrics(init_img, to_tensor(made_image), loss_fn)

                    lpips_list.append(lpips)
                    PSNR_list.append(PSNR)
                    PSNR_nogray_list.append(PSNR_nogray)

                    print(f"{epoch}th epoch - lpips : {lpips}, PSNR : {PSNR}, PSNR_nogray : {PSNR_nogray}")
                    wrtie_to_history(f"\nepoch :{epoch} {idx}th valid img - lpips : {lpips}, PSNR : {PSNR}, PSNR_nogray : {PSNR_nogray}")

                wrtie_to_history(f"\nepoch :{epoch} average lpips : {np.mean(lpips_list)}, PSNR : {np.mean(PSNR_list)}, PSNR_nogray : {np.mean(PSNR_nogray_list)}\n")

                # inf_save(inf_images, img_names, save_path)
                img_names = [f"val {i//2}th img" for i in range(len(inf_images))]
                inf_save(inf_images[:8], img_names[:8], f"./{output_dir}/epoch_{epoch}_valid_images.png")
                inf_save(inf_images[8:], img_names[8:], f"./{output_dir}/epoch_{epoch}_valid_images2.png")
                temp_loss = []
                torch.save(model.state_dict(), output_dir +f"/model_{args.layers_num}_{N_samples}_{epoch}_3d.pt")

            scheduler.step()
    train(model)


if __name__ == "__main__":
    main()
