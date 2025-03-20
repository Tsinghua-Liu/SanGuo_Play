import gradio as gr
from PIL import Image
import os
import json
import sys
import time
from PIL import Image
import base64
from io import BytesIO
import re
import numpy as np
from pathlib import Path
from my_gradio.css_html import create_chat_interface,blockscss
from typing import List, Tuple
from functools import lru_cache
from PIL import ImageOps
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from all_characters.dir_path import CHARACTERS  ## 定义选项字典




def toggle_character(name,selected): #, selected
    if name == None:
        return ""
    if name in selected:
        selected.remove(name)
    else:
        selected.append(name)
    #返回选中的人物列表，用于在界面上显示
    return str(selected),selected

def create_character_selection_interface(selected_characters):
    """
    :param selected_characters: 被选择的角色列表
    :return: 负责根据角色列表，来显示界面
    """
    with gr.Column():
        with gr.Row():
            gr.Markdown("<h1 style='text-align: center; font-size: 48px;'>尔虞我诈是三国</h1>")
            # 显示当前选中的人物
        with gr.Row():
            selected_display = gr.Textbox(label="Selected Characters",value=str(selected_characters.value))

        with gr.Row():
            # 显示每个角色的   图片、名称、选择按钮
            for name, content in CHARACTERS.items():
                with gr.Column():
                    img = Image.open(content["image_path"])
                    button = gr.Button(value=content["zh_name"])
                    button.click(
                        fn=toggle_character,
                        inputs=[gr.Textbox(value=content["zh_name"], visible=False), selected_characters],
                        outputs=[selected_display,selected_characters]
                    )
                    gr.Image(img, elem_id=f"img_{name}", show_label=False)
        with gr.Row():
            switch_to_battle_btn = gr.Button("开始对战")
    return  switch_to_battle_btn,selected_characters


def get_new_md(text: str, image_path: str) -> str:
    """
    根据给定的文字和图片路径，生成一个Markdown块，
    文字位于左侧，图片位于右侧，两者组成的部分居中，无边框。
    """
    img_base64 = optimized_image_processing(image_path)
    # 生成Markdown内容
    md_content = f"""
    <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <div style="margin-right: 10px; font-size: 24px; color: white;">{text}</div>
        <img src="data:image/png;base64,{img_base64}" alt="{text}" style="max-width: 40px; height: 40px; border-radius: 50%;" />
    </div>
    """
    return md_content

@lru_cache(maxsize=128)
def optimized_image_processing(img_path: str,
                               target_size=(40, 40),
                               quality=85) -> str:
    # 优化4：使用更快的缩略图生成方式
    with Image.open(img_path) as img:
        img.thumbnail(target_size)  # 保持比例的快速缩放
        # 转换为RGB模式避免alpha通道影响
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 优化5：使用预分配的BytesIO
        with BytesIO() as buffer:
            # 优化6：降低JPEG质量参数加速编码
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

def add_message(history, img_path, user_input):
    img_str = optimized_image_processing(img_path)
    history.append((img_str, user_input))
    html_c = create_chat_interface(history)
    return history,html_c
