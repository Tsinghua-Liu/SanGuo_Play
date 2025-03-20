import copy
import math
import gradio as gr
from PIL import Image
import os
import json
import sys
import time
import gradio as gr
from PIL import Image
import base64
from io import BytesIO
import re
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple
from functools import lru_cache
from PIL import ImageOps
from my_gradio.ulity import hanzi_to_pinyin
from queue import Queue
from threading import Event
from my_gradio.css_html import create_chat_interface,blockscss
from my_gradio.gradio_module import create_character_selection_interface,toggle_character,add_message,get_new_md
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from all_characters.dir_path import CHARACTERS  ## 定义选项字典
from QA_20.data.classes_generate import CLASS_LIST

# API_MODE = True
# MODEL_PATH = "/mnt/2b44d1e2-569a-42b3-9610-04ea73771f19/liuqinghua/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
#
# MODEL = ""

# 加载背景图片
base_dir = os.path.dirname(os.path.abspath(__file__))
background_image_path = f"{base_dir}/image/back.png"

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

# 创建全局事件和队列
input_event = Event()
USER_INPUT = ""

def interactive_input():
    """模拟input()的阻塞效果"""

    input_event.clear()  # 重置事件
    input_event.wait()  # 阻塞等待

def handle_submit(text):
    global USER_INPUT
    if text.strip():
        USER_INPUT = text
        input_event.set()  # 解除阻塞
        return gr.update(value="")  # 清空输入对话框
    return gr.update()

def process_steps(hosted_value):
    # 步骤1
    if hosted_value:
        print("人工模式")
        interactive_input()
    else:
        print("收到输入")
    return "流程完成！"

# 调整说话者的背景
def back_highlight(name):
    return gr.update(value=get_new_md(name, f"{base_dir}/image/maike.png"))
def back_minimize(name):
    return gr.update(value=f"<div style='text-align: center; font-size: 32px; color: #ff0fff;'>{name}</div>")

def update_all_blocks(name_id_now):
    highlight_number = name_id_now[0]
    updates = []
    for index ,(name, markdown_component) in enumerate(name_blocks_dirt.items()):
        # 更新每个 gr.Markdown 组件的内容
        if index == highlight_number:
            new_value = gr.update(value=get_new_md(name, f"{base_dir}/image/maike.png"))
        else:
            new_value = gr.update(value=f"<div style='text-align: center; font-size: 32px; color: #ff0fff;'>{name}</div>")
        updates.append(new_value)
    return updates
name_blocks_dirt = {}

def start_dialog(secret_word,secret_word_describe, selected_characters , history ,name_id_now,GOOD_ANSWER,hosted,user_input , CONTINUE ):
    global USER_INPUT
    print("进入start",name_id_now)

    zh_name = selected_characters[name_id_now[0]]  #保存正在进行连续对话的角色的姓名
    pinyin_name = hanzi_to_pinyin(zh_name)
    image_path_name = f"{base_dir}/image/{zh_name}.png"
    image_path_judge = f"{base_dir}/image/裁判.png"

    if  GOOD_ANSWER !=  False:
        print("进入start  但是马上退出了  因为已经有正确答案了  所有数据都原路返回，因此不会触发 .change ")
        print(f"start_dialog  GOOD_ANSWER == {GOOD_ANSWER}")
        return history, gr.update(visible=False), name_id_now, GOOD_ANSWER

    if CONTINUE == False :  # 判断结束按键是否 被 按下
        print("进入CONTINUE  但是马上退出了")
        history_ = copy.deepcopy(history)
        history_, html_c = add_message(history_, image_path_judge,
                                       f"游戏被终止")  # 并没有添加到history  但是界面消息会更新这一条。这样就停止了history的无限循环
        if len(history)!=0:
            history = []
            name_id_now = [0,0]
        return history, gr.update(value=html_c), name_id_now, GOOD_ANSWER


    text_history = []  #提取 历史记录当中的文字信息
    for idx, (img_base64, text) in enumerate(history):
        text_history.append(text)

    #角色 生成问题
    if name_id_now[1] ==0:
        name_id_now[1] = name_id_now[1] + 1
        if "用户"  in zh_name and hosted:
            print(zh_name,"进行阻塞")
            interactive_input()
            question = USER_INPUT
        else:
            question = MODEL.llm_generate_character_question(character_name = pinyin_name,describe = secret_word_describe
                                                       , history = text_history,api_model = API_MODE)
        history , html_c = add_message(history, image_path_name, question)
        return history,gr.update(value=html_c),name_id_now,GOOD_ANSWER

    #裁判 生成回复
    elif name_id_now[1] ==1:
        name_id_now[1] = name_id_now[1] + 1
        (img_base64, question) = history[-1]
        answer = MODEL.llm_generate_answer(question = question,keyword = secret_word,api_model = API_MODE)
        history, html_c = add_message(history, image_path_judge, answer)
        return history, gr.update(value=html_c), name_id_now, GOOD_ANSWER

    # 角色 生成猜测
    elif name_id_now[1] == 2:
        name_id_now[1] = name_id_now[1] + 1
        if "用户"  in zh_name and hosted:
            print(zh_name,"进行阻塞")
            interactive_input()
            print(f"用户输入是{USER_INPUT}")
            predict = USER_INPUT
        else:
            predict = MODEL.llm_generate_character_predict(character_name = pinyin_name,describe = secret_word_describe, history = text_history ,api_model = API_MODE)

        history, html_c = add_message(history, image_path_name, predict)
        return history, gr.update(value=html_c), name_id_now, GOOD_ANSWER

    # 裁判 判断对错
    elif name_id_now[1] == 3:


        predict = text_history[-1]
        yesno = MODEL.llm_generate_judge(keyword=secret_word, predict=predict,api_model = API_MODE)
        if "正" in yesno:
            print("回答正确")
            GOOD_ANSWER = True
            # html_c = create_chat_interface([])
            history_ = copy.deepcopy(history)
            history_, html_c = add_message(history_, image_path_judge,
                                           f"{zh_name}   获得最终胜利。 真的好棒啊！")  # 并没有添加到history  但是界面消息会更新这一条。这样就停止了history的无限循环
            history = []
            name_id_now =  [0,0]   # 将  name_id_now   赋值给原来的  State 的 value  所以直接使用列表即可
            return history, gr.update(value=html_c), name_id_now, GOOD_ANSWER

        name_id_now[1] = 0  # 已经执行完一整个角色的流程了，下一个角色应该从头开始
        name_id_now[0] = name_id_now[0] + 1
        if name_id_now[0] == len(selected_characters):  # 已经达到最后一个角色了，应该从头开始
            name_id_now[0] = 0

        history, html_c = add_message(history, image_path_judge, yesno)
        return history, gr.update(value=html_c), name_id_now, GOOD_ANSWER

#用于存储对战界面中，每个人物 的名称 markdown

# 创建Gradio界面
# CSS（Cascading Style Sheets，层叠样式表）是一种用于描述 HTML 或 XML（包括如 SVG、XHTML 等 XML 文档）文档的表现的样式表语言。
# CSS 用于控制网页的布局和外观。

#要输入一个生成模型，来调用它的 generate 方法
with gr.Blocks(
        css=f"""
    .gradio-container {{
        background-image: url('{background_image_path}');
        background-size: cover;
        background-position: center;
    }}
    """) as demo:

    def switch_display():
        return gr.update(visible=True), gr.update(visible=False)

    selected_characters = gr.State(['美女用户', '刘备', '张飞']) #存储选择的姓名
    #selected_characters_textbar = gr.Textbox(value=" ",visible=False,interactive=False)
    history = gr.State([])  # 存储历史对话记录
    name_id_now = gr.State([0, 0])  # 存储当前正在对话的人物姓名
    #name_blocks =  gr.State([])  # 存储姓名对应的markdown组件


    name_blocks = gr.State([])
    # 人物选择界面
    with gr.Row(visible=True) as character_select:
        switch_to_battle_btn,selected_characters = create_character_selection_interface(selected_characters)

    with gr.Row(visible=False) as character_battle:
        with gr.Column():
            with gr.Row():
                @gr.render(inputs=[selected_characters],triggers=[selected_characters.change])
                def generate_dis_of_battle(select_characters):
                    #解除可能存在的阻塞. 这会发生在：  当轮到用户进行输入的时候，系统进入阻塞。此时切换选择人物界面，需要解除阻塞
                    input_event.set()
                    history = gr.State([])  # 存储历史对话记录
                    name_id_now = gr.State([0, 0])  # 存储当前正在对话的人物姓名

                    #print(f"开始创建模型，已经选择武将 为 {selected_characters_text}")

                    print(f"开始创建模型，已经选择武将 为 select_characters {select_characters}")
                    #在这里，selected_characters已经变成了一个普通列表，所以不能再直接作为 State对象进行传递了
                    name_blocks_dirt.clear()

                    GOOD_ANSWER = gr.State(False)
                    CONTINUE = gr.State(True)

                    if len(select_characters) > 0:
                        with gr.Column():
                            #角色显示界面、 对话框显示界面
                            with gr.Row():
                                with gr.Column(scale=1):

                                    for user_name in select_characters[0:math.ceil(len(select_characters) / 2)]:
                                        # 使用正则表达式提取中文字符
                                        chinese_characters = re.findall(r'[\u4e00-\u9fff]+', user_name)
                                        # 将提取出的中文字符列表合并成一个字符串
                                        result = ''.join(chinese_characters)  # 刘备
                                        with gr.Row() as x:
                                            with gr.Column():
                                                with gr.Row():
                                                    current_image = gr.Image(label="图片路径", value=f"{base_dir}/image/{result}.png", height=150, show_label=False,
                                                                             show_share_button=False, width=150, type="filepath",
                                                                             show_download_button=False, show_fullscreen_button=False,container = None)
                                                with gr.Row():
                                                    current_text = gr.Markdown(
                                                        value=f"<div style='text-align: center; font-size: 32px; color: #ff0fff;'>{result}</div>",
                                                        show_label=False)
                                            name_blocks_dirt[result] = current_text

                                with gr.Column(scale=4):
                                    gr.Markdown(
                                        value=f"<div style='text-align: center; font-size: 42px; color: #f00ff;'>{'聊天大厅'}</div>")
                                    html_output = gr.HTML(value=create_chat_interface([]),visible=False)
                                    html_output_dis =  gr.HTML(value=create_chat_interface([]),visible=True)
                                if int(len(select_characters))>=2:
                                    with gr.Column(scale=1):
                                        for user_name in select_characters[math.ceil(len(select_characters) / 2):]:
                                            # 使用正则表达式提取中文字符
                                            chinese_characters = re.findall(r'[\u4e00-\u9fff]+', user_name)
                                            # 将提取出的中文字符列表合并成一个字符串
                                            result = ''.join(chinese_characters)  # 刘备
                                            with gr.Row() as x:
                                                with gr.Column():
                                                    with gr.Row():
                                                        current_image = gr.Image(label="图片路径",
                                                                                 value=f"{base_dir}/image/{result}.png",
                                                                                 height=150, show_label=False,
                                                                                 show_share_button=False, width=150,
                                                                                 type="filepath",
                                                                                 show_download_button=False,
                                                                                 show_fullscreen_button=False)
                                                    with gr.Row():
                                                        current_text = gr.Markdown(
                                                            value=f"<div style='text-align: center; font-size: 32px; color: #ff0fff;'>{result}</div>",
                                                            show_label=False)
                                                name_blocks_dirt[result] = current_text

                            with gr.Row():
                                secret_word = gr.Textbox(value="按下开始会自动初始化", visible=False, interactive=True,label="谜底")
                                secret_word_mask = gr.Textbox(value="**谜底不可见**", visible=True, interactive=True, label="谜底")
                                secret_word_describe = gr.Textbox(value="按下开始会自动初始化", visible=True, interactive=True,label="谜底描述")

                            #对话生成逻辑
                            with gr.Row():

                                start_btn = gr.Button("开始对话")
                                stop_btn = gr.Button("结束对话")
                                #add_btn = gr.Button("添加测试消息")
                                #name_input = gr.Textbox(value="刘备", visible=False)
                                hosted = gr.Checkbox(label="掌管模式（请您输入）", min_width=200,value=True)

                                look_secret = gr.Checkbox(label="查看谜底", min_width=200, value=False)
                                def look_secret_change(checked):
                                    if checked:  #按下  可以查看谜底
                                        return gr.update(visible=False),gr.update(visible=True)
                                    else:
                                        return gr.update(visible=True),gr.update(visible=False)
                                    # 使用 change 事件监听 Checkbox 的状态变化
                                look_secret.change(look_secret_change, inputs=look_secret, outputs=[secret_word_mask,secret_word])

                            with gr.Row():
                                user_input = gr.Textbox(label="请输入", placeholder="输入后按回车继续...")

                                # 设置事件监听
                                user_input.submit(
                                    handle_submit,
                                    inputs=user_input,
                                    outputs=user_input,
                                )

                                def reset_set(GOOD_ANSWER):  #进行开始按钮状态重置
                                    print('\n更新按钮状态************\n' )
                                    print(f"reset_set  GOOD_ANSWER == {GOOD_ANSWER}")
                                    if GOOD_ANSWER == False:
                                        return gr.update(value = "对话已经开始，请您输入")
                                    if GOOD_ANSWER == True:
                                        return gr.update(value = "重新开始")


                                def reset_set_secret():  #进行 new_secret_word 状态重置
                                    new_secret_word = random.sample(CLASS_LIST, 1)
                                    print(' new_secret_word 状态重置')
                                    return gr.update(value=new_secret_word)\
                                        ,gr.update(value=MODEL.llm_generate_keyword(keyword=new_secret_word,api_model = API_MODE))

                                start_btn.click(update_all_blocks, inputs=name_id_now, outputs=list(name_blocks_dirt.values()),show_progress="hidden")\
                                    .then(fn= lambda :True, inputs= None , outputs =CONTINUE)\
                                    .then(fn = lambda :gr.update(value = "对话已经开始，请您输入") , outputs = start_btn)\
                                    .then(fn= lambda : False ,inputs = None,outputs = GOOD_ANSWER)\
                                    .then(fn= reset_set_secret, inputs = None,outputs = [secret_word,secret_word_describe]) \
                                    .then(fn = lambda :gr.update(value=create_chat_interface([])),inputs = None,outputs = html_output_dis)\
                                    .then(fn=start_dialog,
                                        inputs=[secret_word, secret_word_describe,
                                                gr.State(select_characters),
                                                history, name_id_now, GOOD_ANSWER, hosted, user_input,CONTINUE ],
                                        outputs=[history, html_output, name_id_now, GOOD_ANSWER],
                                        show_progress="hidden")
                                # 更新 CONTINUE标志 、更新开始按钮状态、更新GOOD_ANSWER为False、 更新谜底、 更新当前界面、开始循环对话

                                #说明已经结束了
                                GOOD_ANSWER.change(fn = reset_set,
                                                   inputs= GOOD_ANSWER,
                                                    outputs=start_btn)

                                stop_btn.click(fn= lambda :False, inputs= None , outputs =CONTINUE).then(
                                    fn = lambda :gr.update(value = "重新开始") ,
                                    inputs=None,
                                   outputs=[start_btn])\
                                    .then(fn = lambda :gr.update(value=create_chat_interface([])),inputs = None,outputs = html_output_dis)

                        history.change(fn=start_dialog,
                                       inputs=[secret_word, secret_word_describe, gr.State(select_characters), history,
                                               name_id_now, GOOD_ANSWER, hosted, user_input, CONTINUE],
                                       outputs=[history, html_output, name_id_now, GOOD_ANSWER], show_progress="hidden"
                                       )
                        name_id_now.change(fn=(lambda x: x), inputs=html_output, outputs=html_output_dis,
                                           show_progress='hidden')\
                                        .then(update_all_blocks, inputs=name_id_now, outputs=list(name_blocks_dirt.values()),show_progress="hidden")
                    else:
                        print("没有选择人物，无法开始")
                        with gr.Row():
                            gr.Textbox(visible=True, value="没有选择人物，无法开始")
            with gr.Row():
                switch_to_select_btn = gr.Button("选择人物")

    def update_namelist(selected_character):
        if "    " in selected_character:
            selected_character.remove("    ")
        else: selected_character.append("    ")
        return selected_character
    switch_to_battle_btn.click(fn= update_namelist,
                                inputs = [selected_characters],
                                outputs = selected_characters,show_progress="hidden").then(fn= update_namelist,
                                inputs = [selected_characters],
                                outputs = selected_characters,show_progress="hidden").then(fn=switch_display,
                                inputs=None,
                                outputs=[character_battle, character_select])

    switch_to_select_btn.click(fn=switch_display, inputs=None, outputs=[character_select, character_battle])

def generate_gradio(api_mode = True,
    local_model_path = "/mnt/2b44d1e2-569a-42b3-9610-04ea73771f19/liuqinghua/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    ,model = "" ):
    global API_MODE,MODEL_PATH,MODEL

    API_MODE = api_mode
    MODEL_PATH = local_model_path
    MODEL = model


    demo.launch( server_port=7860,share=True)



