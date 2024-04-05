import gradio as gr
import numpy as np
from src.demo.utils import get_point, store_img, get_point_move, store_img_move, clear_points, upload_image_move, segment_with_points, segment_with_points_paste, fun_clear, paste_with_mask_and_offset

examples_remove = [
    [
        "examples/remove/01_moto/0.jpg", # original image 1
        "examples/remove/01_moto/mask0.jpg", # mask 1
        "examples/remove/01_moto/0.jpg", # original image 2
        None, #mask 2
        "examples/remove/01_moto/0.jpg", #Original image 3
        None, #mask 3
        "examples/remove/01_moto/0.jpg", #Original image 4
        "examples/remove/01_moto/mask1.jpg", # refine mask
    ], # 01
    [
        "examples/remove/02_ring/0.jpg", # original image 1
        "examples/remove/02_ring/mask0.jpg", # mask 1
        "examples/remove/02_ring/0.jpg", # original image 2
        "examples/remove/02_ring/mask1.jpg", #mask 2
        "examples/remove/02_ring/0.jpg", #Original image 3
        "examples/remove/02_ring/mask2.jpg", #mask 3
        None, #Original image 4
        None, # refine mask
    ], # 02
    [
        "examples/remove/03_ball/0.jpg", # original image 1
        "examples/remove/03_ball/mask0.jpg", # mask 1
        "examples/remove/03_ball/0.jpg", # original image 2
        "examples/remove/03_ball/mask1.jpg", #mask 2
        "examples/remove/03_ball/0.jpg", #Original image 3
        None, #mask 3
        None, #Original image 4
        None, # refine mask
    ], # 03
    [
        "examples/remove/04_pikachu/0.jpg", # original image 1
        "examples/remove/04_pikachu/mask0.jpg", # mask 1
        "examples/remove/04_pikachu/0.jpg", # original image 2
        "examples/remove/04_pikachu/mask1.jpg", #mask 2
        "examples/remove/04_pikachu/0.jpg", #Original image 3
        "examples/remove/04_pikachu/mask2.jpg", #mask 3
        None, #Original image 4
        None, # refine mask
    ], # 04
    [
        "examples/remove/05_betty/0.jpg", # original image 1
        "examples/remove/05_betty/mask0.jpg", # mask 1
        None, # original image 2
        None, #mask 2
        None, #Original image 3
        None, #mask 3
        None, #Original image 4
        None, # refine mask
    ], # 05
]
examples_zoom = [
    ["examples/zoom/01.jpg"],
    ["examples/zoom/02.jpg"],
    ["examples/zoom/03.jpg"],
    ["examples/zoom/04.jpg"],
    ["examples/zoom/05.jpg"],
    ["examples/zoom/06.jpg"],
    ["examples/zoom/07.jpg"],
]
examples_pan = [
    ["examples/pan/01.jpg"],
    ["examples/pan/02.jpg"],
    ["examples/pan/03.jpg"],
    ["examples/pan/04.jpg"],
    ["examples/pan/05.jpg"],
    ["examples/pan/06.jpg"],
]

examples_moving = [
    [
    "examples/layer/01_horse/00.jpg", #bg
    "examples/layer/01_horse/mask0.jpg", #bg_mask
    0, 0, 1.2, "None", "left/right",  #l1_dx, l1_dy, l1_resize
    ],
    [
    "examples/moving/01_ball/0.jpg", #bg
    "examples/moving/01_ball/mask0.jpg", #bg_mask
    -0.2, -0.1, 0.8, "None", "None",  #l1_dx, l1_dy, l1_resize
    ],
    [
    "examples/moving/02_bell/0.jpg", #bg
    "examples/moving/02_bell/mask0.jpg", #bg_mask
    0, 0, 0.75, "None", "None",  #l1_dx, l1_dy, l1_resize
    ],
]
examples_layer = [
    [
    "examples/layer/01_horse/00.jpg", #bg
    "examples/layer/01_horse/mask0.jpg", #bg_mask

    "examples/layer/01_horse/00.jpg", #l1
    "examples/layer/01_horse/mask0.jpg", #l1_mask
    -0.2, 0, 1, "None", "None", #l1_dx, l1_dy, l1_resize

    "examples/layer/01_horse/00.jpg", #l2
    "examples/layer/01_horse/mask0.jpg", #l2_mask
    0.2, 0, 1, "None", "None", #l2_dx, l2_dy, l2_resize

    None, #l3
    None, #l3_mask
    0, 0, 1, "None", "None", #l3_dx, l3_dy, l3_resize

    "examples/layer/01_horse/00.jpg", #bg_ori
    "examples/layer/01_horse/00.jpg", #l1_ori
    "examples/layer/01_horse/00.jpg", #l2_ori
    None, "None", "None", #l3_ori
    ],

    [
    "examples/layer/02_baby/00.jpg", #bg
    "examples/layer/02_baby/mask0.jpg", #bg_mask

    "examples/layer/02_baby/00.jpg", #l1
    "examples/layer/02_baby/mask1.jpg", #l1_mask
    -0.35, 0, 1,"left/right", "None", #l1_dx, l1_dy, l1_resize

    "examples/layer/02_baby/00.jpg", #l2
    "examples/layer/02_baby/mask2.jpg", #l2_mask
    0.35, 0, 1, "left/right", "None", #l2_dx, l2_dy, l2_resize

    None, #l3
    None, #l3_mask
    0, 0, 1,"None", "None", #l3_dx, l3_dy, l3_resize
    ],

    [
    "examples/layer/03_text/00.jpg", #bg
    "examples/layer/03_text/mask0.jpg", #bg_mask

    "examples/layer/03_text/01.jpg", #l1
    "examples/layer/03_text/mask1.jpg", #l1_mask
    0.1, -0.1, 0.5, "None", "None",#l1_dx, l1_dy, l1_resize

    None, #l2
    None, #l2_mask
    0, 0, 1, "None", "None",#l2_dx, l2_dy, l2_resize

    None, #l3
    None, #l3_mask
    0, 0, 1,"None", "None", #l3_dx, l3_dy, l3_resize
    ],
    [
    "examples/layer/04_cross/0.jpg", #bg
    "examples/layer/04_cross/mask0.jpg", #bg_mask

    "examples/layer/04_cross/2.jpg", #l1
    "examples/layer/04_cross/mask2.jpg", #l1_mask
    -0.1, -0.25, 0.5, "None", "None",#l1_dx, l1_dy, l1_resize

    "examples/layer/04_cross/1.jpg", #l2
    "examples/layer/04_cross/mask1.jpg", #l2_mask
    -0.1, -0.15, 0.7, "None", "None",#l2_dx, l2_dy, l2_resize

    "examples/layer/04_cross/3.jpg", #l3
    "examples/layer/04_cross/mask3.jpg", #l3_mask
    -0.1, -0.55, 0.5, "None", "None",#l3_dx, l3_dy, l3_resize
    ],

]

# 01
def create_demo_remove(runner=None):
    DESCRIPTION = """
    # Object Removal

    ## Usage:

    - Upload a source image, and then draw a box to generate the mask corresponding to the editing object.
    - You can choose to mask more than one object by using Mask2(Draw Box) and Mask3(Sketch).
    - If you encounter artifacts, try to sketch the regions that caused the artifacts.
    - You can refer to the first motorcycle example to understand the usage of the <span style="color:red;">Refined Mask</span>.
    - Please <span style="color:blue;">clear<span> the output before running a new example!
    - For more irregular composition masks, refer to the last page: Mask Preparation.
"""
    
    with gr.Blocks() as demo:
        original_image = gr.State(value=None) 
        img_with_mask = gr.State(value=None) 

        selected_points = gr.State([])
        global_points = gr.State([])
        global_point_label = gr.State([])

        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    # mask 0 
                    gr.Markdown("## Draw box for Mask 1")
                    original_image_1 = gr.Image(source='upload', label="Original image (Mask 1)", interactive=True, type="numpy")
                    # mask 1
                    gr.Markdown("## Option: Draw box for Mask 2")
                    original_image_2 = gr.Image(source='upload', label="Original (Mask 2)", interactive=True, type="numpy")
                    # mask 2
                    gr.Markdown("## Option: Sketch for Mask 3")
                    original_image_3 = gr.Image(tool="sketch", label="Original image (Mask 3)", interactive=True, type="numpy")

                    gr.Markdown("## Option: Mask regions caused artifacts")
                    original_image_4 = gr.Image(tool="sketch", label="Original image (Refine Mask)", interactive=True, type="numpy") 
                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

       
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# Mask")

                    gr.Markdown("## Removal Mask 1")
                    mask_1 = gr.Image(source='upload', label="Removal Mask 1", interactive=True, type="numpy")
                    gr.Markdown("## Option: Removal Mask 2")
                    mask_2 = gr.Image(source='upload', label="Removal Mask 2", interactive=True, type="numpy")
                    gr.Markdown("## Option: Removal Mask 3")
                    mask_3 = gr.Image(source='upload', label="Removal Mask 3", interactive=True, type="numpy")

                    gr.Markdown("## Option: Refine Mask to avoid artifacts:")
                    refine_mask = gr.Image(source='upload', label="Refine Mask", interactive=True, type="numpy")                    
            
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery(columns=1, height='auto')


            original_image_1.select(
                segment_with_points, 
                inputs=[original_image_1, original_image, global_points, global_point_label], 
                outputs=[original_image_1, original_image, mask_1, global_points, global_point_label]
            )
            original_image_2.select(
                segment_with_points, 
                inputs=[original_image_2, original_image, global_points, global_point_label], 
                outputs=[original_image_2, original_image, mask_2, global_points, global_point_label]
            )
            original_image_3.edit(
                store_img_move,
                [original_image_3],
                [original_image, img_with_mask, mask_3]
            )
            original_image_4.edit(
                store_img_move,
                [original_image_4, refine_mask],
                [original_image, img_with_mask, refine_mask]
            )

        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_remove,
                inputs=[
                original_image_1, mask_1, 
                original_image_2, mask_2,
                original_image_3, mask_3, 
                original_image_4, refine_mask]
            )
        run_button.click(fn=runner, inputs=[original_image, mask_1, mask_2, mask_3, refine_mask,
        original_image_1, original_image_2, original_image_3], outputs=[output])
        clear_button.click(
            fn=fun_clear, 
            inputs=[original_image, img_with_mask, selected_points, global_points, global_point_label, original_image_1, original_image_2, original_image_3, original_image_4, mask_1, mask_2, mask_3, refine_mask], 
            outputs=[original_image, img_with_mask, selected_points, global_points, global_point_label, original_image_1, original_image_2, original_image_3, original_image_4, mask_1, mask_2, mask_3, refine_mask]
        )
    return demo


# 02:
def create_demo_zooming(runner=None):
    DESCRIPTION = """
    # Zooming Out

    ## Usage:

    - Upload a source image and choose the width and height zooming scale to zoom out.
    - The illustration of image adjustment and mask preparation is shown in the second column.
    - We recommend setting the zooming scale between <span style="color:red;"> 0.75 <span> and <span style="color:red;"> 1 <span> for optimal results.
    - Please <span style="color:blue;">clear<span> the output before running a new example!
    """
    
    with gr.Blocks() as demo:
        
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    # mask 0
                    gr.Markdown("## Original Image")
                    original_image = gr.Image(source='upload', interactive=True, type="numpy")


                    gr.Markdown("## Scale:") 
                    width_scale= gr.Slider(
                                label="Width scale",
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                value=0.9,
                                interactive=True)
                    height_scale= gr.Slider( 
                                label="Height scale",
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                value=0.9,
                                interactive=True)              
                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# Preprocess")
                    gr.Markdown("## Image Adjustment:")
                    new_image = gr.Gallery(columns=1, height='auto')
                    gr.Markdown("## Mask Adjustment:")
                    new_mask = gr.Gallery(columns=1, height='auto')

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery(columns=1, height='auto')    

        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_zoom,
                inputs=[original_image]
            )
        run_button.click(fn=runner, inputs=[original_image, width_scale, height_scale], outputs=[output, new_image, new_mask])
        clear_button.click(fn=fun_clear, inputs=[original_image, width_scale, height_scale, output, new_image, new_mask], 
        outputs=[original_image, width_scale, height_scale, output, new_image, new_mask])
    return demo
# 03
def create_demo_panning(runner=None):
    DESCRIPTION = """
    # Camera Panning

    ## Usage:

    - Upload a source image and choose the width and height panning scale.
    - The illustration of image adjustment and mask preparation is shown in the second column.
    - We recommend setting the panning scale between<span style="color:red;"> 0 <span> and <span style="color:red;">0.25 <span> for optimal results.
    - Please <span style="color:blue;">clear<span> the output before running a new example!
    """

    with gr.Blocks() as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    # mask 0
                    gr.Markdown("## Original Image")
                    original_image = gr.Image(source='upload', interactive=True, type="numpy")
                    w_direction = gr.Radio(["left", "right"], value="left", label="Width Direction")
                    w_scale = gr.Slider(
                                label="Width scale",
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                value=0,
                                interactive=True)
                    
                    h_direction = gr.Radio(["up", "down"], value="up", label="Height Direction")
                    h_scale = gr.Slider(
                                label="Height scale",
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                value=0,
                                interactive=True)
                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# Preprocess")
                    gr.Markdown("## Image Adjustment:")
                    new_image = gr.Gallery(columns=1, height='auto')
                    gr.Markdown("## Mask Adjustment:")
                    new_mask = gr.Gallery(columns=1, height='auto')

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery(columns=1, height='auto')     

        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_pan,
                inputs=[original_image]
            )
        run_button.click(fn=runner, inputs=[original_image, w_direction, w_scale, h_direction, h_scale], outputs=[output, new_image, new_mask])
        clear_button.click(fn=fun_clear, inputs=[original_image, w_direction, w_scale, h_direction, h_scale, new_image, new_mask, output], 
        outputs=[original_image, w_direction, w_scale, h_direction, h_scale, new_image, new_mask, output])
    return demo
# 04:
def create_position_size(label=None):
    image = gr.Image(source='upload', label=label, interactive=True, type="numpy")
    with gr.Row():
        dx = gr.Slider(
                            label="Left-Right",
                            minimum=-1,
                            maximum=1,
                            step=0.05,
                            value=0,
                            interactive=True
                        )
        dy = gr.Slider(
                            label="Down-Up",
                            minimum=-1,
                            maximum=1,
                            step=0.05,
                            value=0,
                            interactive=True
                        )
    resize_scale = gr.Slider(
                        label="Resize",
                        minimum=0,
                        maximum=2,
                        step=0.05,
                        value=1,
                        interactive=True
                    )
    with gr.Row():
        w_flip = gr.Radio(["left/right","None"], value="None", label="Horizontal Flip")
        h_flip = gr.Radio(["down/up", "None"], value="None", label="Vertical Flip")
    return image, dx, dy, resize_scale, w_flip, h_flip
# 05:
def create_demo_layer(runner=None):
    DESCRIPTION = """
    # 🚩 Multi-Layered Editing 🚩

    ## Usage:

    - Notice that all operations can be achieved using the multi-layered editing mode.
    - In particular, you can accomplish multi-object editing such as adding objects and cross-image composition on this page.
    - Try some interesting examples given below to understand the usage.
    - Please <span style="color:blue;">clear<span> the output before running a new example!
    - We strongly recommend you to read the [original paper](https://arxiv.org/abs/2403.14487) to further explore more uses of multi-layered editing.
    """
    selected_points = gr.State([])
    global_points = gr.State([])
    global_point_label = gr.State([])
    bg_ori = gr.State(value=None)
    l1_ori = gr.State(value=None)
    l2_ori = gr.State(value=None)
    l3_ori = gr.State(value=None)
    with gr.Blocks() as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## Background Image")
                    bg_img = gr.Image(source='upload', label="Background", interactive=True, type="numpy")
                    gr.Markdown("## Layer-1")
                    l1_img, l1_dx, l1_dy, l1_resize, l1_w_flip, l1_h_flip = create_position_size(label="Layer-1")
                    gr.Markdown("## Layer-2")
                    l2_img, l2_dx, l2_dy, l2_resize, l2_w_flip, l2_h_flip = create_position_size(label="Layer-2")
                    gr.Markdown("## Layer-3")
                    l3_img, l3_dx, l3_dy, l3_resize, l3_w_flip, l3_h_flip = create_position_size(label="Layer-3")
                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# Mask")
                    gr.Markdown("## Background Mask for Removal:")
                    bg_mask =  gr.Image(source='upload', label="BG Mask", interactive=True, type="numpy")
                    gr.Markdown("## Layer-1 Mask:")
                    l1_mask = gr.Image(source='upload', label="L1 Mask", interactive=True, type="numpy")
                    gr.Markdown("## Layer-2 Mask:")
                    l2_mask = gr.Image(source='upload', label="L2 Mask", interactive=True, type="numpy")
                    gr.Markdown("## Layer-3 Mask:")
                    l3_mask = gr.Image(source='upload', label="L3 Mask", interactive=True, type="numpy")

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery(columns=1, height='auto')    

        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")            
            gr.Examples(
                examples=examples_layer,
                inputs=[
                bg_img, bg_mask,
                l1_img, l1_mask, l1_dx, l1_dy, l1_resize, l1_w_flip, l1_h_flip,
                l2_img, l2_mask, l2_dx, l2_dy, l2_resize, l2_w_flip, l2_h_flip,
                l3_img, l3_mask, l3_dx, l3_dy, l3_resize, l3_w_flip, l3_h_flip,
                ]
            )
        bg_img.select(
                segment_with_points, 
                inputs=[bg_img, bg_ori, global_points, global_point_label], 
                outputs=[bg_img, bg_ori, bg_mask, global_points, global_point_label]
        )
        l1_img.select(
                segment_with_points, 
                inputs=[l1_img, l1_ori, global_points, global_point_label], 
                outputs=[l1_img, l1_ori, l1_mask, global_points, global_point_label]
        )
        l2_img.select(
                segment_with_points, 
                inputs=[l2_img, l2_ori, global_points, global_point_label], 
                outputs=[l2_img, l2_ori, l2_mask, global_points, global_point_label]
        )
        l3_img.select(
                segment_with_points, 
                inputs=[l3_img, l3_ori, global_points, global_point_label], 
                outputs=[l3_img, l3_ori, l3_mask, global_points, global_point_label]
        )

        run_button.click(fn=runner, inputs=[
        bg_img, 
        l1_img, l1_dx, l1_dy, l1_resize, l1_w_flip, l1_h_flip, 
        l2_img, l2_dx, l2_dy, l2_resize, l2_w_flip, l2_h_flip, 
        l3_img, l3_dx, l3_dy, l3_resize, l3_w_flip, l3_h_flip,
        bg_mask, l1_mask, l2_mask, l3_mask,
        bg_ori, l1_ori, l2_ori, l3_ori
        ], outputs=[output])

        clear_button.click(fn=fun_clear, 
        inputs=[bg_img, bg_ori, 
        l1_img, l1_ori, l1_dx, l1_dy, l1_resize, l1_w_flip, l1_h_flip,
        l2_img, l2_ori, l2_dx, l2_dy, l2_resize, l2_w_flip, l2_h_flip,
        l3_img, l3_ori, l3_dx, l3_dy, l3_resize, l3_w_flip, l3_h_flip,
        bg_mask, l1_mask, l2_mask, l3_mask,
        global_points, global_point_label, output],
        outputs=[bg_img, bg_ori, 
        l1_img, l1_ori, l1_dx, l1_dy, l1_resize, l1_w_flip, l1_h_flip,
        l2_img, l2_ori, l2_dx, l2_dy, l2_resize, l2_w_flip, l2_h_flip,
        l3_img, l3_ori, l3_dx, l3_dy, l3_resize, l3_w_flip, l3_h_flip,
        bg_mask, l1_mask, l2_mask, l3_mask,
        global_points, global_point_label, output],            
        )
    return demo

# 06:
def create_demo_mask_box(runner=None):
    DESCRIPTION = """
    # 🔧 Mask Preparation 
    ## Usage:
    - This page is a tool for you to combine more than one mask.
    - You can draw a box to mask an object to obtain Masks 1-3, and sketch to obtain Mask 4.
    - The merged mask is the union of Masks 1-4.
    - Please <span style="color:blue;">clear<span> the output before running a new example!
    """
    
    with gr.Blocks() as demo:
        original_image = gr.State(value=None) 
        img_with_mask = gr.State(value=None)
        selected_points = gr.State([])
        global_points = gr.State([])
        global_point_label = gr.State([])
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## 1. Draw box for Mask 1")
                    img_draw_box_1 = gr.Image(source='upload', label="Original Image", interactive=True, type="numpy")

                    gr.Markdown("## 2. Draw box for Mask 2")
                    img_draw_box_2 = gr.Image(source='upload', label="Original Image", interactive=True, type="numpy")

                    gr.Markdown("## 3. Draw box for Mask 3")
                    img_draw_box_3 = gr.Image(source='upload', label="Original Image", interactive=True, type="numpy")

                    gr.Markdown("## 4. Sketch for Mask 4")
                    img_sketch_4 = gr.Image(tool="sketch", label="Original Image", interactive=True, type="numpy")

                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# Mask")
                    gr.Markdown("## Mask 1")
                    mask_1 = gr.Image(source='upload', label="Mask", interactive=True, type="numpy")
                    gr.Markdown("## Mask 2")
                    mask_2 = gr.Image(source='upload', label="Mask", interactive=True, type="numpy")
                    gr.Markdown("## Mask 3")
                    mask_3 = gr.Image(source='upload', label="Mask", interactive=True, type="numpy")
                    gr.Markdown("## Mask 4")
                    mask_4 = gr.Image(source='upload', label="Mask", interactive=True, type="numpy")

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# Merged Mask")
                    merged_mask = gr.Image(source='upload', label="Mask of object", interactive=True, type="numpy")  

            img_draw_box_1.select(
                segment_with_points, 
                inputs=[img_draw_box_1, original_image, global_points, global_point_label], 
                outputs=[img_draw_box_1, original_image, mask_1, global_points, global_point_label]
            )
            img_draw_box_2.select(
                segment_with_points, 
                inputs=[img_draw_box_2, original_image, global_points, global_point_label], 
                outputs=[img_draw_box_2, original_image, mask_2, global_points, global_point_label]
            )
            img_draw_box_3.select(
                segment_with_points, 
                inputs=[img_draw_box_3, original_image, global_points, global_point_label], 
                outputs=[img_draw_box_3, original_image, mask_2, global_points, global_point_label]
            )
            img_sketch_4.edit(
                store_img_move,
                [img_sketch_4],
                [original_image, img_with_mask, mask_4]
            )

        run_button.click(fn=runner, inputs=[mask_1, mask_2, mask_3, mask_4], outputs=[merged_mask])
        clear_button.click(
        fn=fun_clear, 
        inputs=[original_image, img_with_mask, selected_points, global_points, global_point_label, img_draw_box_1, img_draw_box_2, img_draw_box_3, img_sketch_4, mask_1, mask_2, mask_3, mask_4], 
        outputs=[original_image, img_with_mask, selected_points, global_points, global_point_label, img_draw_box_1, img_draw_box_2, img_draw_box_3, img_sketch_4, mask_1, mask_2, mask_3, mask_4, merged_mask]
    )
    return demo

def create_demo_moving(runner=None):
    DESCRIPTION = """
    # Object Moving, Resizing, and Flipping

    ## Usage:
    - Upload an image and draw a box around the object to manipulate.
    - Move the object vertically or horizontally using sliders or by drawing an arrow.
    - You can select options for moving and flipping the object from a menu.
    - Please <span style="color:blue;">clear<span> the output before running a new example!
    """

    selected_points = gr.State([])
    global_points = gr.State([])
    global_point_label = gr.State([])
    bg_ori = gr.State(value=None)
    l1_ori = gr.State(value=None)
    with gr.Blocks() as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## Draw box to mask target object")
                    bg_img = gr.Image(source='upload', label="Background", interactive=True, type="numpy")
                    gr.Markdown("## Draw arrow to describe the movement")
                    l1_img, l1_dx, l1_dy, l1_resize, l1_w_flip, l1_h_flip = create_position_size(label="Layer-1")
                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# Mask")
                    gr.Markdown("## Background Mask for Removal:")
                    bg_mask =  gr.Image(source='upload', label="Mask", interactive=True, type="numpy")

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery(columns=1, height='auto')  

        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_moving,
                inputs=[
                bg_img, bg_mask, l1_dx, l1_dy, l1_resize, l1_h_flip, l1_w_flip
                ]
            )
        bg_img.select(
                segment_with_points, 
                inputs=[bg_img, bg_ori, global_points, global_point_label], 
                outputs=[bg_img, bg_ori, bg_mask, global_points, global_point_label]
        )
        l1_img.select(
                get_point_move,
                [bg_ori, l1_img, selected_points],
                [l1_img, bg_ori, selected_points, l1_dx, l1_dy],
        )

        run_button.click(fn=runner, inputs=[
        bg_img, bg_ori,bg_mask, 
        l1_dx, l1_dy, l1_resize, l1_w_flip, l1_h_flip, selected_points
        ], outputs=[output])

        clear_button.click(fn=fun_clear, 
        inputs=[bg_img, bg_ori, bg_mask, l1_img, l1_ori, l1_dx, l1_dy, l1_resize, l1_w_flip, l1_h_flip,
        global_points, global_point_label, selected_points, output],
        outputs=[bg_img, bg_ori, bg_mask, l1_img, l1_ori, l1_dx, l1_dy, l1_resize, l1_w_flip, l1_h_flip,
        global_points, global_point_label, selected_points, output],         
        )
    return demo
