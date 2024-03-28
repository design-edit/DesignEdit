import numpy as np
import torch
from diffusers import  DDIMScheduler
import cv2
from utils.sdxl import sdxl
from utils.inversion import Inversion
import math
import torch.nn.functional as F
import utils.utils as utils
import os 
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

MAX_NUM_WORDS = 77

def init_model(model_path, model_dtype="fp16", num_ddim_steps=50):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    if model_dtype == "fp16":
        torch_dtype = torch.float16
    elif model_dtype == "fp32":
        torch_dtype = torch.float32

    pipe = sdxl.from_pretrained(model_path, torch_dtype=torch_dtype, use_safetensors=True, variant=model_dtype,scheduler=scheduler)
    pipe.to(device)
    inversion = Inversion(pipe,num_ddim_steps)
    return pipe, inversion

class LayerFusion:   
    def get_mask(self, maps, alpha, use_pool,x_t):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = F.interpolate(maps, size=(x_t.shape[2:])) #[2, 1, 128, 128]
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask=(mask - mask.min ()) / (mask.max () - mask.min ())
        mask = mask.gt(self.mask_threshold)
        self.mask=mask
        mask = mask[:1] + mask
        return mask 

    def get_one_mask(self, maps, use_pool, x_t, idx_lst, i=None, sav_img=False):
        k=1
        if sav_img is False:
            mask_tot = 0
            for obj in idx_lst:
                mask = maps[0, :, :, :, obj].mean(0).reshape(1, 1, 32, 32)
                if use_pool:
                    mask = F.max_pool2d(mask, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
                mask = F.interpolate(mask, size=(x_t.shape[2:]))
                mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
                mask=(mask - mask.min ()) / (mask.max () - mask.min ())
                mask = mask.gt(self.mask_threshold[int(self.counter/10)])
                mask_tot |= mask
            mask = mask_tot  
            return mask
        else: 
            for obj in idx_lst:
                mask = maps[0, :, :, :, obj].mean(0).reshape(1, 1, 32, 32)
                if use_pool:
                    mask = F.max_pool2d(mask, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
                mask = F.interpolate(mask, size=(1024, 1024))#[1, 1, 1024, 1024]
                mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
                mask=(mask - mask.min ()) / (mask.max () - mask.min ())
                mask = mask.gt(0.6)  
                mask = np.array(mask[0][0].clone().cpu()).astype(np.uint8)*255
                cv2.imwrite(f'./img/sam_mask/{self.blend_list[i][0]}_{self.counter}.jpg', mask)
        return mask

    def mv_op(self, mp, op, scale=0.2, ones=False, flip=None):
        _, b, H, W = mp.shape
        if ones == False:
            new_mp = torch.zeros_like(mp)
        else:
            new_mp = torch.ones_like(mp)
        K = int(scale*W)
        if op == 'right':
            new_mp[:, :, :, K:] = mp[:, :, :, 0:W-K]
        elif op == 'left':
            new_mp[:, :, :, 0:W-K] = mp[:, :, :, K:]
        elif op == 'down':
            new_mp[:, :, K:, :] = mp[:, :, 0:W-K, :]
        elif op == 'up':
            new_mp[:, :, 0:W-K, :] = mp[:, :, K:, :]
        if flip is not None:
            new_mp = torch.flip(new_mp, dims=flip)
               
        return new_mp

    def mv_layer(self, x_t, bg_id, fg_id, op_id):
        bg_img = x_t[bg_id:(bg_id+1)].clone()
        fg_img = x_t[fg_id:(fg_id+1)].clone()
        fg_mask = self.fg_mask_list[fg_id-3]
        op_list = self.op_list[fg_id-3]

        for item in op_list:
            op, scale = item[0], item[1]
            if scale != 0:
                fg_img = self.mv_op(fg_img, op=op, scale=scale)
                fg_mask = self.mv_op(fg_mask, op=op, scale=scale)
        x_t[op_id:(op_id+1)] = bg_img*(1-fg_mask) + fg_img*fg_mask

    def __call__(self, x_t):
        self.counter += 1
        # inpainting
        if self.blend_time[0] <= self.counter <= self.blend_time[1]:
            x_t[1:2] = x_t[1:2]*self.remove_mask + x_t[0:1]*(1-self.remove_mask) 

        if self.counter == self.blend_time[1] + 1 and self.mode != "removal":
            b = x_t.shape[0]
            bg_id = 1 #bg_layer
            op_id = 2 #canvas
            for fg_id in range(3, b): #fg_layer
                self.mv_layer(x_t, bg_id=bg_id, fg_id=fg_id, op_id=op_id)
                bg_id = op_id
    
        return x_t

    def __init__(self, remove_mask, fg_mask_list, refine_mask=None, 
                blend_time=[0, 40],
                 mode="removal", op_list=None):
        self.counter = 0
        self.mode = mode
        self.op_list = op_list
        self.blend_time = blend_time

        self.remove_mask = remove_mask
        self.refine_mask = refine_mask
        if self.refine_mask is not None:
            self.new_mask = self.remove_mask + self.refine_mask
            self.new_mask[self.new_mask>0] = 1
        else:
            self.new_mask = None
        self.fg_mask_list = fg_mask_list


class Control():
    def step_callback(self, x_t):
        if self.layer_fusion is not None:
             x_t = self.layer_fusion(x_t)
        return x_t
    def __init__(self, layer_fusion):
        self.layer_fusion = layer_fusion

def register_attention_control(model, controller, mask_time=[0, 40], refine_time=[0, 25]):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        self.counter = 0 #time
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None): #self_attention
            x = hidden_states.clone() 
            context = encoder_hidden_states
            is_cross = context is not None
            if is_cross is False:
                if controller.layer_fusion is not None and (mask_time[0] < self.counter < mask_time[1]):
                    b, i, j = x.shape
                    H = W = int(math.sqrt(i))
                    x_old = x.clone()
                    x = x.reshape(b, H, W, j)
                    new_mask = controller.layer_fusion.remove_mask
                    if new_mask is not None:
                        new_mask[new_mask>0] = 1
                        new_mask = F.interpolate(new_mask.to(dtype=torch.float32).clone(), size=(H, W), mode='bilinear').cuda()
                        new_mask =  (1 - new_mask).reshape(1, H, W).unsqueeze(-1)
                        if (refine_time[0] < self.counter <= refine_time[1]) and controller.layer_fusion.refine_mask is not None:
                            new_mask = controller.layer_fusion.new_mask
                            new_mask = F.interpolate(new_mask.to(dtype=torch.float32).clone(), size=(H, W), mode='bilinear').cuda()
                            new_mask =  (1 - new_mask).reshape(1, H, W).unsqueeze(-1)                
                        idx = 1 #inpaiint_idx:bg
                        x[int(b/2)+idx, :, :] = (x[int(b/2)+idx, :, :]*new_mask[0])
                    x = x.reshape(b, i, j)
            if is_cross: 
                q = self.to_q(x) 
                k = self.to_k(context)
                v = self.to_v(context)
            else:
                context = x
                q = self.to_q(hidden_states) 
                k = self.to_k(x) 
                v = self.to_v(hidden_states)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            if hasattr(controller, 'count_layers'):
                controller.count_layers(place_in_unet,is_cross)
            sim = torch.einsum("b i d, b j d -> b i j", q.clone(), k.clone()) * self.scale 

            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
            global global_cnt
            self.counter += 1
            return to_out(out)
        
        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

class DesignEdit():
    def __init__(self, pretrained_model_path="/home/jyr/model/stable-diffusion-xl-base-1.0"):
        self.model_dtype = "fp16"
        self.pretrained_model_path=pretrained_model_path
        self.num_ddim_steps = 50
        self.mask_time = [0, 40]
        self.op_list = {}
        self.attend_scale = {}
        self.ldm_model, self.inversion= init_model(model_path=self.pretrained_model_path, model_dtype=self.model_dtype, num_ddim_steps=self.num_ddim_steps)
    
    def run_remove(self, original_image=None, mask_1=None, mask_2=None, mask_3=None, refine_mask=None, 
        ori_1=None, ori_2=None, ori_3=None,
        prompt="", save_dir="./tmp", mode='removal',):
        # 01-1: 
        if original_image is None:
            original_image = ori_1 if ori_1 is not None else ori_2 if ori_2 is not None else ori_3
        op_list = None
        attend_scale = 20
        sample_ref_match={0 : 0, 1 : 0}
        ori_shape = original_image.shape

        # 01-2: prepare: image_gt, remove_mask, fg_mask_list, refine_mask
        image_gt = Image.fromarray(original_image).resize((1024, 1024))
        image_gt = np.stack([np.array(image_gt)])
        mask_list = [mask_1, mask_2, mask_3]
        remove_mask = utils.attend_mask(utils.add_masks_resized(mask_list), attend_scale=attend_scale) # numpy to tensor
        fg_mask_list = None
        refine_mask = utils.attend_mask(utils.convert_and_resize_mask(refine_mask)) if refine_mask is not None else None

        # 01-3: prepare: prompts, blend_time, refine_time
        prompts = len(sample_ref_match)*[prompt] # 2
        blend_time = [0, 41]
        refine_time = [0, 25]
        
        # 02: invert
        _, x_t, x_stars, prompt_embeds, pooled_prompt_embeds = self.inversion.invert(image_gt, prompts, inv_batch_size=1)
        
        # 03: init layer_fusion and controller
        lb = LayerFusion(remove_mask=remove_mask, fg_mask_list=fg_mask_list, refine_mask=refine_mask,
                    blend_time=blend_time, mode=mode, op_list=op_list)
        controller = Control(layer_fusion=lb)
        register_attention_control(model=self.ldm_model, controller=controller, mask_time=self.mask_time, refine_time=refine_time)
        
        # 04: generate images
        images = self.ldm_model(controller=controller, prompt=prompts,
                        latents=x_t, x_stars=x_stars,  
                        negative_prompt_embeds=prompt_embeds, 
                        negative_pooled_prompt_embeds=pooled_prompt_embeds,
                        sample_ref_match=sample_ref_match)
        folder = None
        utils.view_images(images, folder=folder)
        return [cv2.resize(images[1], (ori_shape[1], ori_shape[0]))]


    def run_zooming(self, original_image, width_scale=1, height_scale=1, prompt="", save_dir="./tmp", mode='removal'):
        # 01-1: 
        op_list = {0: ['zooming', [height_scale, width_scale]]}
        ori_shape = original_image.shape
        attend_scale = 30
        sample_ref_match = {0 : 0, 1 : 0}

        # 01-2: prepare: image_gt, remove_mask, fg_mask_list, refine_mask
        img_new, mask = utils.zooming(original_image, [height_scale, width_scale])
        img_new_copy = img_new.copy()
        mask_copy = mask.copy()
        
        image_gt = Image.fromarray(img_new).resize((1024, 1024))
        image_gt = np.stack([np.array(image_gt)])

        remove_mask = utils.attend_mask(utils.convert_and_resize_mask(mask), attend_scale=attend_scale) # numpy to tensor
        fg_mask_list = None
        refine_mask = None

        # 01-3: prepare: prompts, blend_time, refine_time
        prompts = len(sample_ref_match)*[prompt] # 2
        blend_time = [0, 41]
        refine_time = [0, 25]

        # 02: invert
        _, x_t, x_stars, prompt_embeds, pooled_prompt_embeds = self.inversion.invert(image_gt, prompts, inv_batch_size=1)
        
        # 03: init layer_fusion and controller
        lb = LayerFusion(remove_mask=remove_mask, fg_mask_list=fg_mask_list, blend_time=blend_time,
                    mode=mode, op_list=op_list)
        controller = Control(layer_fusion=lb)
        register_attention_control(model=self.ldm_model, controller=controller, mask_time=self.mask_time, refine_time=refine_time)
        
        # 04: generate images
        images = self.ldm_model(controller=controller, prompt=prompts,
                        latents=x_t, x_stars=x_stars,  
                        negative_prompt_embeds=prompt_embeds, 
                        negative_pooled_prompt_embeds=pooled_prompt_embeds,
                        sample_ref_match=sample_ref_match)
        folder = None
        utils.view_images(images, folder=folder)
        resized_img = cv2.resize(images[1], (ori_shape[1], ori_shape[0]))
        return [resized_img], [img_new_copy], [mask_copy]

    def run_panning(self, original_image, w_direction, w_scale, h_direction, h_scale, prompt="", save_dir="./tmp", mode='removal'):
        # 01-1: prepare: op_list, attend_scale, sample_ref_match
        ori_shape = original_image.shape
        attend_scale = 30
        sample_ref_match = {0 : 0, 1 : 0}

        # 01-2: prepare: image_gt, remove_mask, fg_mask_list, refine_mask
        op_list = [[w_direction, w_scale], [h_direction, h_scale]]
        img_new, mask = utils.panning(original_image, op_list=op_list)
        img_new_copy = img_new.copy()
        mask_copy = mask.copy()
        
        image_gt = Image.fromarray(img_new).resize((1024, 1024))
        image_gt = np.stack([np.array(image_gt)])
        remove_mask = utils.attend_mask(utils.convert_and_resize_mask(mask), attend_scale=attend_scale) # numpy to tensor

        fg_mask_list = None
        refine_mask = None

        # 01-3: prepare: prompts, blend_time, refine_time
        prompts = len(sample_ref_match)*[prompt] # 2
        blend_time = [0, 41]
        refine_time = [0, 25]

        # 02: invert
        _, x_t, x_stars, prompt_embeds, pooled_prompt_embeds = self.inversion.invert(image_gt, prompts, inv_batch_size=1)
        # 03: init layer_fusion and controller
        lb = LayerFusion(remove_mask=remove_mask, fg_mask_list=fg_mask_list, blend_time=blend_time,
                    mode=mode, op_list=op_list)
        controller = Control(layer_fusion=lb)
        register_attention_control(model=self.ldm_model, controller=controller, mask_time=self.mask_time, refine_time=refine_time)
        
        # 04: generate images
        images = self.ldm_model(controller=controller, prompt=prompts,
                        latents=x_t, x_stars=x_stars,  
                        negative_prompt_embeds=prompt_embeds, 
                        negative_pooled_prompt_embeds=pooled_prompt_embeds,
                        sample_ref_match=sample_ref_match)
        folder = None
        utils.view_images(images, folder=folder)
        resized_img = cv2.resize(images[1], (ori_shape[1], ori_shape[0]))
        return [resized_img], [img_new_copy], [mask_copy]

    # layer-wise multi-object editing
    def process_layer_states(self, layer_states):
        image_paths = []
        mask_paths = []
        op_list = []
        
        for state in layer_states:
            img, mask, dx, dy, resize, w_flip, h_flip = state
            if img is not None:  
                img = cv2.resize(img, (1024, 1024))
                mask = utils.convert_and_resize_mask(mask)
                dx_command = ['right', dx] if dx > 0 else ['left', -dx]
                dy_command = ['up', dy] if dy > 0 else ['down', -dy]
                flip_code = None
                if w_flip == "left/right" and h_flip == "down/up":
                    flip_code = -1
                elif w_flip == "left/right":
                    flip_code = 1  # 或者其他默认值，根据您的需要设置
                elif h_flip == "down/up":
                    flip_code = 0
                op_list.append([dx_command, dy_command])
                img, mask, _ = utils.resize_image_with_mask(img, mask, resize)
                img, mask, _ = utils.flip_image_with_mask(img, mask, flip_code=flip_code)
                image_paths.append(img)
                mask_paths.append(utils.attend_mask(mask))
        sample_ref_match = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3}
        required_length = len(image_paths) + 3
        truncated_sample_ref_match = {k: sample_ref_match[k] for k in sorted(sample_ref_match.keys())[:required_length]}
        return image_paths, mask_paths, op_list, truncated_sample_ref_match


    def run_layer(self, bg_img, l1_img, l1_dx, l1_dy, l1_resize, l1_w_flip, l1_h_flip, 
        l2_img, l2_dx, l2_dy, l2_resize, l2_w_flip, l2_h_flip,
        l3_img, l3_dx, l3_dy, l3_resize, l3_w_flip, l3_h_flip,
        bg_mask, l1_mask, l2_mask, l3_mask,
        bg_ori=None, l1_ori=None, l2_ori=None, l3_ori=None,
        prompt="", save_dir="./tmp", mode='layerwise'):
        # 00： prepare: layer-wise states
        bg_img = bg_ori if bg_ori is not None else bg_img
        l1_img = l1_ori if l1_ori is not None else l1_img
        l2_img = l2_ori if l2_ori is not None else l2_img
        l3_img = l3_ori if l3_ori is not None else l3_img
        for mask in [bg_mask, l1_mask, l2_mask, l3_mask]:
            if mask is None:
                mask = np.zeros((1024, 1024), dtype=np.uint8)
            else:
                mask = utils.convert_and_resize_mask(mask)
        l1_state = [l1_img, l1_mask, l1_dx, l1_dy, l1_resize, l1_w_flip, l1_h_flip]
        l2_state = [l2_img, l2_mask, l2_dx, l2_dy, l2_resize, l2_w_flip, l2_h_flip]
        l3_state = [l3_img, l3_mask, l3_dx, l3_dy, l3_resize, l3_w_flip, l3_h_flip]
        ori_shape = bg_img.shape

        image_paths, fg_mask_list, op_list, sample_ref_match = self.process_layer_states([l1_state, l2_state, l3_state])
        if image_paths == []:
            mode = "removal"
        # 01-1: prepare: image_gt, remove_mask, fg_mask_list, refine_mask
        attend_scale = 20
        image_gt = [bg_img] + image_paths
        image_gt = [Image.fromarray(img).resize((1024, 1024)) for img in image_gt]
        image_gt = np.stack(image_gt)      
        remove_mask = utils.attend_mask(bg_mask, attend_scale=attend_scale)
        refine_mask = None

        # 01-2: prepare: promptrun_masks, blend_time, refine_time
        prompts = len(sample_ref_match)*[prompt] # 2
        blend_time = [0, 41]
        refine_time = [0, 25]
        attend_scale = []

        # 02: invert
        _, x_t, x_stars, prompt_embeds, pooled_prompt_embeds = self.inversion.invert(image_gt, prompts, inv_batch_size=len(image_gt))
        # 03: init layer_fusion and controller
        lb = LayerFusion(remove_mask=remove_mask, fg_mask_list=fg_mask_list, blend_time=blend_time, refine_mask=refine_mask,
                    mode=mode, op_list=op_list)
        controller = Control(layer_fusion=lb)
        register_attention_control(model=self.ldm_model, controller=controller, mask_time=self.mask_time, refine_time=refine_time)
        # 04: generate images
        images = self.ldm_model(controller=controller, prompt=prompts,
                        latents=x_t, x_stars=x_stars,  
                        negative_prompt_embeds=prompt_embeds, 
                        negative_pooled_prompt_embeds=pooled_prompt_embeds,
                        sample_ref_match=sample_ref_match)
        folder = None
        utils.view_images(images, folder=folder) 
        if mode == 'removal':
            resized_img = cv2.resize(images[1], (ori_shape[1], ori_shape[0]))       
        else:
            resized_img = cv2.resize(images[2], (ori_shape[1], ori_shape[0]))       
        return [resized_img]
    

    def run_moving(self, bg_img, bg_ori, bg_mask, l1_dx, l1_dy, l1_resize, 
    l1_w_flip=None, l1_h_flip=None, selected_points=None,
        prompt="", save_dir="./tmp", mode='layerwise'):
        # 00： prepare: layer-wise states
        bg_img = bg_ori if bg_ori is not None else bg_img
        l1_img = bg_img
        if bg_mask is None:
            bg_mask = np.zeros((1024, 1024), dtype=np.uint8)
        else:
            bg_mask = utils.convert_and_resize_mask(bg_mask)
        l1_mask = bg_mask
        l1_state = [l1_img, l1_mask, l1_dx, l1_dy, l1_resize, l1_w_flip, l1_h_flip]
        ori_shape = bg_img.shape

        image_paths, fg_mask_list, op_list, sample_ref_match = self.process_layer_states([l1_state])

        # 01-1: prepare: image_gt, remove_mask, fg_mask_list, refine_mask
        attend_scale = 20
        image_gt = [bg_img] + image_paths
        image_gt = [Image.fromarray(img).resize((1024, 1024)) for img in image_gt]
        image_gt = np.stack(image_gt)      
        remove_mask = utils.attend_mask(bg_mask, attend_scale=attend_scale)
        refine_mask = None

        # 01-2: prepare: promptrun_masks, blend_time, refine_time
        prompts = len(sample_ref_match)*[prompt] # 2
        blend_time = [0, 41]
        refine_time = [0, 25]
        attend_scale = []

        # 02: invert
        _, x_t, x_stars, prompt_embeds, pooled_prompt_embeds = self.inversion.invert(image_gt, prompts, inv_batch_size=len(image_gt))
        # 03: init layer_fusion and controller
        lb = LayerFusion(remove_mask=remove_mask, fg_mask_list=fg_mask_list, blend_time=blend_time, refine_mask=refine_mask,
                    mode=mode, op_list=op_list)
        controller = Control(layer_fusion=lb)
        register_attention_control(model=self.ldm_model, controller=controller, mask_time=self.mask_time, refine_time=refine_time)
        # 04: generate images
        images = self.ldm_model(controller=controller, prompt=prompts,
                        latents=x_t, x_stars=x_stars,  
                        negative_prompt_embeds=prompt_embeds, 
                        negative_pooled_prompt_embeds=pooled_prompt_embeds,
                        sample_ref_match=sample_ref_match)
        folder = None
        utils.view_images(images, folder=folder) 
        resized_img = cv2.resize(images[2], (ori_shape[1], ori_shape[0]))       
        return [resized_img]

    # turn mask to 1024x1024 unit-8
    def run_mask(self, mask_1, mask_2, mask_3, mask_4):
        mask_list = [mask_1, mask_2, mask_3, mask_4]
        final_mask = utils.add_masks_resized(mask_list)
        return final_mask