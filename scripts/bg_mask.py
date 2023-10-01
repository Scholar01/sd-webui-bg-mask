import gradio as gr
import rembg
from modules import scripts
from modules.ui_components import InputAccordion, FormRow

models = [
    "None",
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
]


class BackgroundMaskScript(scripts.Script):
    # 对图片进行放大扩充的脚本
    id_part = "bg_mask"

    def __init__(self):
        super().__init__()
        self.mask = None

    def title(self):
        return "Background Mask"

    def show(self, is_img2img):
        if is_img2img:
            return scripts.AlwaysVisible
        else:
            return False

    def ui(self, is_img2img):
        id_part = self.id_part
        with gr.Accordion(label=self.title(), elem_id=f"{id_part}_main_accordion",open=False):
            with FormRow():
                model = gr.Dropdown(choices=models, value="u2net", show_label=False)
                enable_i2irembg = gr.Checkbox(label="Enable", value=False)
                return_mask = gr.Checkbox(label="Return mask", value=False)
                alpha_matting = gr.Checkbox(label="Alpha matting", value=False)

            with FormRow(visible=False) as alpha_mask_row:
                alpha_matting_erode_size = gr.Slider(label="Erode size", minimum=0, maximum=40, step=1, value=10)
                alpha_matting_foreground_threshold = gr.Slider(label="Foreground threshold", minimum=0, maximum=255,
                                                               step=1, value=240)
                alpha_matting_background_threshold = gr.Slider(label="Background threshold", minimum=0, maximum=255,
                                                               step=1, value=10)

            alpha_matting.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[alpha_matting],
                outputs=[alpha_mask_row],
            )
        return [enable_i2irembg, model, return_mask, alpha_matting, alpha_matting_erode_size,
                alpha_matting_foreground_threshold, alpha_matting_background_threshold]

    def before_process(self, p, *args):
        """
        预处理
        """
        enable_i2irembg, model, return_mask, alpha_matting, alpha_matting_erode_size, alpha_matting_foreground_threshold, alpha_matting_background_threshold, *more = args
        if not enable_i2irembg:
            return

        if not model or model == "None":
            return

        image = p.init_images[0]
        # image = images.resize_image(p.resize_mode, image, p.width, p.height)

        mask = rembg.remove(
            image,
            session=rembg.new_session(model),
            only_mask=True,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
        )
        self.mask = mask

        p.image_mask = mask
        p.mask_blur_x = 1
        p.mask_blur_y = 1
        p.inpaint_full_res_padding = 32  # 预留像素

        p.inpainting_mask_invert = 0  # 蒙版模式 0=重绘蒙版内容
        p.inpainting_fill = 1  # 蒙版蒙住的内容 0=填充
        p.inpaint_full_res = 0  # 重绘区域 0 = 全图 1=仅蒙版区域

    def postprocess(self, p, processed, *args):
        enable_i2irembg, model, return_mask, alpha_matting, alpha_matting_erode_size, alpha_matting_foreground_threshold, alpha_matting_background_threshold, *more = args

        if not enable_i2irembg:
            return

        if not model or model == "None":
            return

        if not return_mask:
            return

        processed.images.append(self.mask)
