import torch
import numpy as np
from PIL import Image
from typing import Optional, Union, Tuple, List
from tqdm import tqdm
import os
from diffusers import DDIMInverseScheduler,DPMSolverMultistepInverseScheduler
class Inversion:

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    @torch.no_grad()
    def get_noise_pred_single(self, latents, t, context,cond=True,both=False):
        added_cond_id=1 if cond else 0
        do_classifier_free_guidance=False
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        if both is False:
            added_cond_kwargs = {"text_embeds": self.add_text_embeds[added_cond_id].unsqueeze(0).repeat(self.inv_batch_size,1), "time_ids": self.add_time_ids[added_cond_id].unsqueeze(0).repeat(self.inv_batch_size,1)}
        else:
            added_cond_kwargs = {"text_embeds": self.add_text_embeds, "time_ids": self.add_time_ids}
        noise_pred = self.model.unet(
            latent_model_input,
            t,
            encoder_hidden_states=context,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        return noise_pred

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / self.model.vae.config.scaling_factor * latents.detach()
        self.model.vae.to(dtype=torch.float32)
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            else:
                if image.ndim==3:
                    image=np.expand_dims(image,0)
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(0, 3, 1, 2).to(self.device)
                latents=[]
                for i,_ in enumerate(image):
                    latent=self.model.vae.encode(image[i:i+1])['latent_dist'].mean
                    latents.append(latent)
                latents=torch.stack(latents).squeeze(1)
                latents = latents * self.model.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def init_prompt(
        self,
        prompt:  Union[str, List[str]],
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
    ):
        original_size = original_size or (1024, 1024)
        target_size = target_size or (1024, 1024)
        # 3. Encode input prompt
        do_classifier_free_guidance=True
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.model.encode_prompt_not_zero_uncond(
            prompt,
            self.model.device,
            1,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
        )
        prompt_embeds=prompt_embeds[:self.inv_batch_size]
        negative_prompt_embeds=negative_prompt_embeds[:self.inv_batch_size]
        pooled_prompt_embeds=pooled_prompt_embeds[:self.inv_batch_size]
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds[:self.inv_batch_size]
        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self.model._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        self.add_text_embeds = add_text_embeds.to(self.device)
        self.add_time_ids = add_time_ids.to(self.device).repeat(self.inv_batch_size * 1, 1)

        self.prompt_embeds=prompt_embeds
        self.negative_prompt_embeds=negative_prompt_embeds
        self.pooled_prompt_embeds=pooled_prompt_embeds
        self.negative_pooled_prompt_embeds=negative_pooled_prompt_embeds
        self.prompt = prompt
        self.context=prompt_embeds

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(self.generator, self.eta)
        if isinstance(self.inverse_scheduler,DDIMInverseScheduler):
            extra_step_kwargs.pop("generator")
        for i in tqdm(range(self.num_ddim_steps)):
            use_inv_sc=False 
            if use_inv_sc:
                t = self.inverse_scheduler.timesteps[i]
                noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings,cond=True)
                latent = self.inverse_scheduler.step(noise_pred, t, latent, **extra_step_kwargs, return_dict=False)[0]
            else:
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
                noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings,cond=True)
                latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image) 
        image_rec = self.latent2image(latent) 
        ddim_latents = self.ddim_loop(latent.to(self.model.unet.dtype)) 
        return image_rec, ddim_latents

    from typing import Union, List, Dict
    import numpy as np

    def invert(self, image_gt, prompt: Union[str, List[str]], 
            verbose=True, inv_output_pos=None, inv_batch_size=1):

        self.inv_batch_size = inv_batch_size
        self.init_prompt(prompt)
        out_put_pos = 0 if inv_output_pos is None else inv_output_pos
        self.out_put_pos = out_put_pos
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Done.")
        return (image_gt, image_rec), ddim_latents[-1], ddim_latents, self.prompt_embeds[self.prompt_embeds.shape[0]//2:], self.pooled_prompt_embeds


    def __init__(self, model,num_ddim_steps,generator=None,scheduler_type="DDIM"):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.num_ddim_steps=num_ddim_steps
        if scheduler_type == "DDIM":
            self.inverse_scheduler=DDIMInverseScheduler.from_config(self.model.scheduler.config)
            self.inverse_scheduler.set_timesteps(num_ddim_steps)
        elif scheduler_type=="DPMSolver":
            self.inverse_scheduler=DPMSolverMultistepInverseScheduler.from_config(self.model.scheduler.config)
            self.inverse_scheduler.set_timesteps(num_ddim_steps)
        self.model.scheduler.set_timesteps(num_ddim_steps)
        self.model.vae.to(dtype=torch.float32)
        self.prompt = None
        self.context = None
        self.device=self.model.unet.device
        self.generator=generator
        self.eta=0.0

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def load_1024_mask(image_path, left=0, right=0, top=0, bottom=0,target_H=128,target_W=128):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, np.newaxis]
    else:
        image = image_path
    if len(image.shape) == 4:
        image = image[:, :, :, 0]
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image=image.squeeze()
    image = np.array(Image.fromarray(image).resize((target_H, target_W)))
    return image

def load_1024(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).resize((1024, 1024)))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((1024, 1024)))
    return image