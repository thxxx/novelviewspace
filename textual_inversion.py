import argparse
import os
import lpips
from datetime import datetime
import random
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import matplotlib.pyplot as plt
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from diffusers import StableDiffusionPipeline
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from torch import autocast
from utils import inf_save, get_metrics, process_img
from torchvision.transforms.functional import to_pil_image
import torchvision
import json

logger = get_logger(__name__)
loss_fn = lpips.LPIPS(net='alex')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        required=False,
        help="Path to input image to edit.",
    )
    parser.add_argument(
        "--source_text",
        type=str,
        default=None,
        help="The source text describing the input image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument("--initial", type=str, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--n_hiper",
        type=int,
        default=5,
        help="Number of hiper embedding",
    )
    parser.add_argument(
        "--emb_train_steps",
        type=int,
        default=1500,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--emb_train_epochs",
        type=int,
        default=4,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--emb_learning_rate",
        type=float,
        default=3e-3,
        help="Learning rate for optimizing the embeddings.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_false", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open("valids_hotdog.json", "r") as jso:
        validation_datas = json.load(jso)
    for valid_idx, valid_data in enumerate(validation_datas):
        if valid_idx<2:
            continue
        # Setting
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
        )

        if args.seed is not None:
            set_seed(args.seed)
            g_cuda = torch.Generator(device='cuda')
            g_cuda.manual_seed(args.seed)

        os.makedirs(args.output_dir, exist_ok=True)
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.Adam8bit
        else:
            optimizer_class = torch.optim.Adam
        # optimizer_class = torch.optim.RMSprop

        weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Load pretrained models
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name, subfolder="tokenizer", use_auth_token=True)
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name, subfolder="unet", use_auth_token=True)
        noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        # Encode the input image.
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name, subfolder="vae", use_auth_token=True)
        CLIP_text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name, subfolder="text_encoder", use_auth_token=True)
        vae.to(accelerator.device, dtype=weight_dtype)

        init_image, input_image = process_img("dataset/hotdogs/" + valid_data['img_path'], args.resolution)
        init_image = init_image[None].to(device=accelerator.device, dtype=weight_dtype)
        input_image = input_image.squeeze()
        input_image = to_pil_image(input_image)

        with torch.inference_mode():
            init_latents = vae.encode(init_image).latent_dist.sample()
            init_latents = 0.18215 * init_latents

        # Encode the source and target text.
        CLIP_text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_ids_tgt = tokenizer("", padding="max_length", truncation=True,
                                    max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
        text_ids_tgt = text_ids_tgt.to(device=accelerator.device)

        #! It should be changed to dataset
        text_ids_src = tokenizer(args.source_text, padding="max_length", truncation=True,
                                    max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
        text_ids_src = text_ids_src.to(device=accelerator.device)
        with torch.inference_mode():
            target_embeddings = CLIP_text_encoder(text_ids_tgt)[0].float()
            source_embeddings = CLIP_text_encoder(text_ids_src)[0].float()

        # del vae, CLIP_text_encoder
        del vae, CLIP_text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # For inference
        ddim_scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, steps_offset=1, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        pipe = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name, scheduler=ddim_scheduler, torch_dtype=torch.float16, safety_checker = None).to("cuda")
        num_samples = 1
        guidance_scale = 7.5
        num_inference_steps = 50
        height = 512
        width = 512

        # Optimize hiper embedding
        n_hipers = [args.n_hiper]
        learning_rates = [args.emb_learning_rate]

        for lr in learning_rates:
            for n_hiper in n_hipers:
                losses = []
                train_history={
                    "valid_data":valid_data,
                    "lpips":[],
                    "psnr":[],
                }
                hiper_embeddings = source_embeddings[:, -n_hiper:].clone().detach()
                if args.initial:
                    print("\n\nload : ", hiper_embeddings.shape)
                    hiper_embeddings = torch.load(args.initial).clone().detach().to("cuda")
                    # hiper_embeddings=hiper_embeddings.to(accelerator.device)
                    print("load : ", hiper_embeddings, "\n\n")
                src_embeddings = source_embeddings[:, :-n_hiper].clone().detach()
                tgt_embeddings = target_embeddings[:, :-n_hiper].clone().detach()
                hiper_embeddings.requires_grad_(True)

                optimizer = optimizer_class(
                    [hiper_embeddings],
                    lr=lr,
                    betas=(args.adam_beta1, args.adam_beta2),
                    eps=args.adam_epsilon,
                )

                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

                unet, optimizer = accelerator.prepare(unet, optimizer)

                def train_loop(pbar, optimizer, hiper_embeddings, epoch):
                    temp_loss = []
                    for step in pbar:
                        with accelerator.accumulate(unet):
                            noise = torch.randn_like(init_latents)
                            bsz = init_latents.shape[0]
                            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=init_latents.device)
                            timesteps = timesteps.long()

                            noisy_latents = noise_scheduler.add_noise(init_latents, noise, timesteps)
                            
                            source_embeddings = torch.cat([src_embeddings, hiper_embeddings], 1)
                            noise_pred = unet(noisy_latents, timesteps, source_embeddings).sample
                            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                            temp_loss.append(loss.cpu().detach())

                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

                        if step != 0 and step % 50 == 0:
                            losses.append((sum(temp_loss)/50).item())
                            temp_loss = []

                        # Check inference
                        if step != 0 and step % (args.emb_train_steps-1) == 0:
                            plt.clf()
                            plt.plot(losses)
                            plt.ylim(0, 0.1)
                            plt.savefig(args.output_dir + f'/losses_{valid_idx}.png')

                            loss = round((sum(losses)/len(losses)), 10)
                            record_result(hiper_embeddings, epoch, loss)

                    accelerator.wait_for_everyone()

                def record_result(hiper_embeddings, epoch, loss, seed_fix=False):
                    file_path = f'{args.output_dir}/result_{valid_idx}.txt'
                    PSNR_list = []
                    PSNR1_list = []
                    lpips_list = []
                    for i in range(5):
                        inf_emb = []
                        inf_emb.append(
                            torch.cat([src_embeddings, hiper_embeddings.clone().detach()], 1))

                        inf_images = []
                        inf_images.append(input_image)
                        seed = random.randrange(1, 10000)
                        with autocast("cuda"), torch.inference_mode():
                            for embs in inf_emb:
                                set_seed(seed)
                                g_cuda = torch.Generator(device='cuda')
                                g_cuda.manual_seed(seed)
                                pass_nsfw=False
                                images = pipe(prompt_embeds=embs, height=height, width=width, num_images_per_prompt=num_samples,
                                                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=g_cuda)
                                for j in range(len(images.images)):
                                    inf_images.append(images.images[j])
                                    break
                        
                        # input_image랑 inf_images[-1]이랑 비교
                        totensor = torchvision.transforms.ToTensor()
                        lpips, PSNR, PSNR1 = get_metrics(init_image, totensor(inf_images[-1]), loss_fn)
                        print(f"lpips : {lpips}, PSNR : {PSNR}, PSNR1 : {PSNR1}")
                        PSNR_list.append(PSNR)
                        PSNR1_list.append(PSNR1)
                        lpips_list.append(lpips)

                        with open(file_path, 'a') as file:
                            file.write("\n" + f"---{i}th image seed : {seed} lpips : {lpips}, PSNR : {PSNR} , PSNR1 : {PSNR1}")
                    
                        save_name = os.path.join(args.output_dir, f'{args.source_text}_epoch_{epoch}_{valid_idx}_{seed}.png')
                        inf_save(inf_images, [f'loss:{loss}', f'[{args.source_text}, hper]', '[empty, hper]'], save_name)
                        del images

                        if seed_fix:
                            break

                    embedding_name = os.path.join(args.output_dir, f'{args.source_text}__{valid_idx}_nhiper{n_hiper}.pt')
                    torch.save(hiper_embeddings.cpu(), embedding_name)

                    train_history['lpips'].append(np.mean(lpips_list))
                    train_history['psnr'].append(np.mean(PSNR1_list))

                    # save quantative results
                    with open(file_path, 'a') as file:
                        file.write(f"\n valid index : {valid_idx} " + str(datetime.now()) + f"\ndataset_{args.source_text}, epoch_{epoch}, n_hiper_{n_hiper}, learning_rate_{optimizer.param_groups[0]['lr']} loss_{loss}")
                        file.write("\n" + f"len : {len(lpips_list)} , lpips : {np.mean(lpips_list)}, PSNR : {np.mean(PSNR_list)} , PSNR1 : {np.mean(PSNR1_list)} \n")
                    
                    with open(f'{args.output_dir}/save.json', "a") as jso:
                        json.dump(train_history, jso)

                for i in range(0, args.emb_train_epochs):
                    progress_bar = tqdm(range(args.emb_train_steps), disable=not accelerator.is_local_main_process)
                    progress_bar.set_description("Optimizing embedding")
                    print("\nlr : ", optimizer.param_groups[0]['lr'], "\n")
                    train_loop(progress_bar, optimizer, hiper_embeddings, i)
                    # if i>4:
                    scheduler.step()

if __name__ == "__main__":
    main()

