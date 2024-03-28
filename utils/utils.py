import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import os
from typing import List, Dict

def convert_and_resize_mask(mask):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    resized_mask = cv2.resize(mask, (1024, 1024))       
    return resized_mask

def add_masks_resized(masks):
    final_mask = np.zeros((1024, 1024), dtype=np.uint8)         
    for mask in masks:
        if mask is not None:
            resized_mask = convert_and_resize_mask(mask)
            resized_mask = resized_mask.astype(np.uint8)
            final_mask = cv2.add(final_mask, resized_mask)
    return final_mask

def attend_mask(mask_file, attend_scale=10, save=False):
    if isinstance(mask_file, str):
        if mask_file == '':
            return torch.zeros([1, 1, 128, 128], dtype=torch.float32).cuda()
        else:
            image_with_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    elif len(mask_file.shape) == 3: # convert RGB to gray
        image_with_mask = cv2.cvtColor(mask_file, cv2.COLOR_BGR2GRAY)
    
    else:
        image_with_mask = mask_file

    if attend_scale != 0:
        kernel = np.ones((abs(attend_scale), abs(attend_scale)), np.uint8)        
        if attend_scale > 0:
            image_with_mask = cv2.dilate(image_with_mask, kernel, iterations=1)
        else:
            image_with_mask = cv2.erode(image_with_mask, kernel, iterations=1)
        
        if save and isinstance(mask_file, str):
            new_mask_file_name = mask_file[:-4]+'_'+str(attend_scale)+'.jpg'
            cv2.imwrite(new_mask_file_name, image_with_mask)
            print("new_mask is saved in ", new_mask_file_name)

    dilated_image= cv2.resize(image_with_mask, (128, 128), interpolation=cv2.INTER_NEAREST)
    dilated_image = torch.from_numpy(dilated_image).to(torch.float32).unsqueeze(0).unsqueeze(0).cuda() / 255 
    return dilated_image


def panning(img_path=None, op_list=[['left', 0.2]], save=False, save_dir=None):
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    img_new = img.copy()
    img_height, img_width, _ = img.shape
    w_mask = 255 * np.ones((img_height, img_width), dtype=np.uint8)
    h_mask = 255 * np.ones((img_height, img_width), dtype=np.uint8)

    for op in op_list:
        scale = op[1]
        if op[0] in ['right', 'left']:
            K = int(scale*img_width)
        elif op[0] in ['up', 'down']:
            K = int(scale*img_height)
      
        if op[0] == 'right':
            img_new[:, K:, :] = img[:, 0:img_width-K, :]
            w_mask[:, K:] = 0
        elif op[0] == 'left':
            img_new[:, 0:img_width-K, :] = img[:, K:, :]
            w_mask[:, 0:img_width-K] = 0
        elif op[0] == 'down':
            img_new[K:, :, :] = img[0:img_height-K, :, :]
            h_mask[K:, :] = 0
        elif op[0] == 'up':
            img_new[0:img_height-K, :, :] = img[K:, :, :]
            h_mask[0:img_height-K, :] = 0
        img = img_new
    
    mask = w_mask + h_mask
    mask[mask>0] = 255
    
    if save:
        if save_dir is None:
            base_dir = os.path.dirname(img_path)
            save_dir = os.path.join(base_dir, 'preprocess')
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        resized_img_name = f"{save_dir}/resized_image.png"
        resized_mask_name = f"{save_dir}/resized_mask.png"
        cv2.imwrite(resized_img_name, img_new)
        cv2.imwrite(resized_mask_name, mask)
        return resized_img_name, resized_mask_name
    else:
        return img_new, mask

def zooming(img_path=None, scale=[0.8, 0.8], save=False, save_dir=None):
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    img_new = img.copy()
    img_height, img_width, _ = img.shape
    mask = 255 * np.ones((img_height, img_width), dtype=np.uint8)

    new_height = int(img_height*scale[0])
    new_width = int(img_width*scale[1])
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    x_offset = (img_width - new_width) // 2
    y_offset = (img_height - new_height) // 2

    img_new[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
    mask[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = 0

    if save:
        if save_dir is None:
            base_dir = os.path.dirname(img_path)
            save_dir = os.path.join(base_dir, 'preprocess')
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)

        resized_img_name = f"{save_dir}/resized_image.png"
        resized_mask_name = f"{save_dir}/resized_mask.png"
        cv2.imwrite(resized_img_name, img_new)
        cv2.imwrite(resized_mask_name, mask)
        return resized_img_name, resized_mask_name
    else:
        return img_new, mask

def get_box(mask, bias = 2):
    nonzero_indices = torch.nonzero(mask)
    H, W = mask.shape[-2:]
    min_x = max(min(nonzero_indices[:, 1]) - bias, 0)
    min_y = max(min(nonzero_indices[:, 0]) - bias, 0)
    max_x = min(max(nonzero_indices[:, 1]) + bias, W)
    max_y = min(max(nonzero_indices[:, 0]) + bias, H)
    return (min_x, min_y, max_x, max_y)


def draw_axis(img,grid_dict,x_len,y_len):
    if grid_dict is not None and grid_dict is not False:
        assert isinstance(grid_dict,Dict)
        assert "x_title" in grid_dict
        assert "y_title" in grid_dict
        assert "x_text_list" in grid_dict
        assert "y_text_list" in grid_dict
        x_title=grid_dict["x_title"]
        y_title=grid_dict["y_title"]
        x_text_list=grid_dict['x_text_list']
        y_text_list=grid_dict['y_text_list']
        assert len(y_text_list)==y_len
        assert len(x_text_list)==x_len
        assert "font_size" in grid_dict
        font_size=grid_dict["font_size"]
        if "x_color" in grid_dict:
            color_x=grid_dict['x_color']
        else:
            color_x="black"
        if "y_color" in grid_dict:
            color_y=grid_dict['y_color']
        else:
            color_y="black"
        if "num_decimals" in grid_dict:
            num_decimals=grid_dict['num_decimals']
        else:
            num_decimals=2
        if "shift_x" in grid_dict:
            shift_x_x,shift_x_y=grid_dict['shift_x']
        else:
            shift_x_x=shift_x_y=0
        if "shift_y" in grid_dict:
            shift_y_x,shift_y_y=grid_dict['shift_y']
        else:
            shift_y_x=shift_y_y=0
        if "title" in grid_dict:
            title=grid_dict['title']
            if isinstance(title,List):
                all_title=""
                for s in title:
                    all_title=all_title+s+"\n"
                title=all_title
        else:
            title=''
        width, height = img.size
        num_x=x_len
        num_y=y_len

        new_img = Image.new("RGB", (width + width // num_x+width // (num_x*2), height + height // num_y+height // (num_y*2)), color=(255, 255, 255))
        width,height=(width + width // num_x, height + height // num_y)
        num_x=num_x+1
        num_y=num_y+1
        new_img.paste(img, (width // num_x, height // num_y))

        draw = ImageDraw.Draw(new_img)

        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
        for i in range(2, num_x+1):
            x = (i - 1) * width // num_x + width // (num_x * 2)-width *0.2// num_x+shift_x_x
            y = height // (num_y * 2)+shift_x_y
            k=i-1
            if  isinstance(x_text_list[i-2],str):
                draw.text((x, y), x_text_list[i-2], font=font,fill=color_x,align="center")
            else:
                draw.text((x, y), "{:.{}f}".format(x_text_list[i-2],num_decimals), font=font,fill=color_x,align="center")

        for i in range(2, num_y+1):
            x = width // (num_x * 2)-width *0.1// num_x+shift_y_x
            y = (i - 1) * height // num_y + height // (num_y * 2)-height*0.1//num_y+shift_y_y
            k = i - 1
            if isinstance(y_text_list[i-2],str):
                draw.text((x, y), y_text_list[i-2], font=font,fill=color_y,align="center")
            else:
                draw.text((x, y), "{:.{}f}".format(y_text_list[i-2],num_decimals), font=font,fill=color_y,align="center")
        i=1
        x = (i - 1) * width // num_x + width // (num_x * 2)-height*0.1//num_y+shift_y_x
        y = height // (num_y * 2)+width *0.2// num_x+shift_y_y
        draw.text((x, y), y_title, font=font, fill=color_y,align="center")
        x = width // (num_x * 2)+width *0.2// num_x+shift_x_x
        y = (i - 1) * height // num_y + height // (num_y * 2)+shift_x_y
        draw.text((x, y), x_title, font=font, fill=color_x,align="left")
        x = width // 4
        y = (i - 1) * height // num_y + height // (num_y * 10)
        draw.text((x, y), title, font=font, fill='blue',align="left")
    else:

        new_img=img
    return new_img

def view_images(images, num_rows=1, offset_ratio=0.02,text="",folder=None,Notimestamp=False,
grid_dict=None,subfolder=None,verbose=True,output_dir=None,timestamp=None,**kwargs):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0
    origin_size=kwargs.get("origin_size",None)
    images_copy=images.copy()
    for i, per_image in enumerate(images_copy):
        if isinstance(per_image, Image.Image) and origin_size is not None:
            images[i] = np.array(per_image.resize((origin_size[1],origin_size[0])))
        else:
            images[i] = np.array(per_image)
        
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

    pil_img_=draw_axis(pil_img,grid_dict,num_cols,num_rows)
    if pil_img_.size[0]==pil_img_.size[1]:
        pil_img_.resize((2048,2048))
    else:
        longer_side = max(pil_img.size)
        ratio = 2048/longer_side
        new_size = tuple([int(x*ratio) for x in pil_img.size])
        pil_img = pil_img.resize(new_size)

    if verbose is False:
        return pil_img
    now = datetime.now()
    if timestamp is None:
        if Notimestamp is False:
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            timestamp=""
    if output_dir is None:
        if timestamp != "":
            date, time = timestamp.split('_')
        else:
            date, time = "",""
        if folder is not None:
            dirname="./"+folder
            filename = text+f"img_{timestamp}.jpg"
        else:
            if subfolder is not None:
                dirname=os.path.join("./img", subfolder,date)
                dirname=os.path.join(dirname,time)            
                filename =text+f"img_{timestamp}.jpg"
            else:
                dirname=os.path.join("./img",date)
                dirname=os.path.join(dirname,time)
                filename =text+f"img_{timestamp}.jpg"
    else:
        dirname=output_dir
        filename =text+f"img_{timestamp}.jpg"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if verbose is True:
        for i, img in enumerate(images):
            im = Image.fromarray(img)
            im.save(os.path.join(dirname,f"{i}.jpg"))
    print(f"Output dir: {dirname}")
    pil_img.save(os.path.join(dirname, filename))
    if grid_dict is not None and grid_dict is not False:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        pil_img_.save(os.path.join(dirname, filename[:-4]+"_2048x.jpg"))

def resize_image_with_mask(img, mask, scale):
    if scale == 1:
        return img, mask, None
    img_blackboard = img.copy() # canvas
    mask_blackboard = np.zeros_like(mask)

    M = cv2.moments(mask)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    scale_factor = [scale, scale]
    resized_img = cv2.resize(img, None, fx=scale_factor[0], fy=scale_factor[1], interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, None, fx=scale_factor[0], fy=scale_factor[1], interpolation=cv2.INTER_AREA)
    new_cx, new_cy = cx * scale_factor[0], cy * scale_factor[1]

    for y in range(resized_mask.shape[0]):
        for x in range(resized_mask.shape[1]):
            if 0 <= cy - (new_cy - y) < img.shape[0] and 0 <= cx - (new_cx - x) < img.shape[1]:
                mask_blackboard[int(cy - (new_cy - y)), int(cx - (new_cx - x))] = resized_mask[y, x]
                img_blackboard[int(cy - (new_cy - y)), int(cx - (new_cx - x))] = resized_img[y, x]
    return img_blackboard, mask_blackboard, (cx, cy)

def flip_image_with_mask(img, mask, flip_code=None):
    if flip_code is None:
        return img, mask, None
    M = cv2.moments(mask)
    if M["m00"] == 0:  
        return img, mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    h, w = img.shape[:2]
    img_center = (w // 2, h // 2)

    tx = img_center[0] - cx
    ty = img_center[1] - cy

    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    img_translated = cv2.warpAffine(img, M_translate, (w, h))
    mask_translated = cv2.warpAffine(mask, M_translate, (w, h))
    flipped_img = cv2.flip(img_translated, flip_code)
    flipped_mask = cv2.flip(mask_translated, flip_code)
    M_translate_back = np.float32([[1, 0, -tx], [0, 1, -ty]])
    flipped_img_back = cv2.warpAffine(flipped_img, M_translate_back, (w, h))
    flipped_mask_back = cv2.warpAffine(flipped_mask, M_translate_back, (w, h))

    return flipped_img_back, flipped_mask_back, (cx, cy)
