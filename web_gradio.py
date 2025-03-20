from QA_20.QA_20_2_1 import Qwen7BModel
from all_characters.dir_path import CHARACTERS  ## 定义选项字典
from my_gradio.character_select2_1 import *
import argparse



def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Run the Qwen7B model with specified parameters.")

    # 添加命令行参数
    parser.add_argument("--api_mode", type=bool, default=True, help="Enable or disable API mode (default: True).")
    parser.add_argument("--model_path", type=str,default=r"D:\model\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28",  help="Path to the model directory.")

    # 解析命令行参数
    args = parser.parse_args()

    # 获取参数值
    api_mode = args.api_mode
    model_path = args.model_path

    # 定义选项字典
    characters_list = CHARACTERS

    # 初始化模型
    qwen7b_model = Qwen7BModel(characters_list, model_path, api_mode)

    # 调用 Gradio 函数
    generate_gradio(api_mode=api_mode, model=qwen7b_model)


if __name__ == "__main__":
    main()


