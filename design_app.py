import os
import subprocess
import shlex
from src.demo.model import DesignEdit

os.makedirs('models', exist_ok=True)
subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/model/efficient_sam_vits.pt -O models/efficient_sam_vits.pt'))

from src.demo.demo import *
import shlex

import cv2
import gradio as gr
pretrained_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
model =  DesignEdit(pretrained_model_path=pretrained_model_path)
DESCRIPTION_1 = """<div style="text-align: center; font-size: 80px;">
        <strong class="title is-1">
            <span style="color: green;">🌿D</span>
            <span style="color: orange;">e</span>
            <span style="color: rgb(63, 185, 63);">s</span>
            <span style="color: green;">i</span>
            <span style="color: rgb(200, 85, 23);">g</span>
            <span style="color: green;">n</span>
            <span style="color: orange;">E</span>
            <span style="color: crimson;">d</span>
            <span style="color: darkorange;">i</span>
            <span style="color: green;">t🌿</span>
          </strong> 
    </div>
    """
DESCRIPTION_2 = """ <div style="text-align: center;font-size: 24px;"> <h1> Multi-Layered Latent Decomposition and Fusion for Unified & Accurate Image Editing</h1></div>"""
DESCRIPTION_3 = """
<div style="text-align: center; font-size: 24px;">
    <p> Gradio demo for <a href="https://design-edit.github.io/">DesignEdit</a></p>
</div>
"""

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION_1)
    gr.Markdown(DESCRIPTION_2)
    gr.Markdown(DESCRIPTION_3)
    with gr.Tabs():
        with gr.TabItem('1️⃣ Object Removal'):
            create_demo_remove(model.run_remove)
        with gr.TabItem('2️⃣ Zooming Out'):
            create_demo_zooming(model.run_zooming)
        with gr.TabItem('3️⃣ Camera Panning'):
            create_demo_panning(model.run_panning)
        with gr.TabItem('4️⃣ Object Moving, Resizing and Flipping'):
            create_demo_moving(model.run_moving)
        with gr.TabItem('5️⃣ 🚩 Multi-Layered Editing 🚩'):
            create_demo_layer(model.run_layer)
        with gr.TabItem('🔧 Mask Preparation: Draw or Sketch'):
            create_demo_mask_box(model.run_mask)
demo.queue(concurrency_count=3, max_size=20)
demo.launch(server_name="0.0.0.0")

