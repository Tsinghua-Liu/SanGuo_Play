import os
import re


def find_max_checkpoint(folder_path):
    """
    找到文件夹中数字中间的的 checkpoint-num 文件名。

    参数:
    folder_path (str): 文件夹路径

    返回:
    str: 数字最大的 checkpoint 文件名，如果没有找到则返回 None
    """
    # 正则表达式匹配 "checkpoint-num" 格式的文件名
    pattern = re.compile(r"checkpoint-(\d+)")

    # 存储所有匹配的文件名和对应的数字
    checkpoints = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))  # 提取数字部分
            checkpoints.append((num, filename))

    # 找到数字最大的文件名
    if checkpoints:
        max_checkpoint = min(checkpoints, key=lambda x: x[0])
        return max_checkpoint[1]
    else:
        return None

abs_path = os.getcwd() #os.path.abspath(__file__)

CHARACTERS = {
    "zhangfei": {"image_path": f"{abs_path}/my_gradio/image/张飞.png",
    "lora_model": f"{abs_path}/lora/output/Qw2.5_7B_lora_zhangfei/{find_max_checkpoint(f'{abs_path}/lora/output/Qw2.5_7B_lora_zhangfei')}",
    "zh_name":"张飞"
  },
"caocao": {"image_path": f"{abs_path}/my_gradio/image/曹操.png",
    "lora_model": f"{abs_path}/lora/output/Qw2.5_7B_lora_caocao/{find_max_checkpoint(f'{abs_path}/lora/output/Qw2.5_7B_lora_caocao')}",
    "zh_name":"曹操"
  },
"liubei": {"image_path": f"{abs_path}/my_gradio/image/刘备.png",
    "lora_model": f"{abs_path}/lora/output/Qw2.5_7B_lora_liubei/{find_max_checkpoint(f'{abs_path}/lora/output/Qw2.5_7B_lora_liubei')}",
    "zh_name":"刘备"
  },
"meinvyonghu": {"image_path": f"{abs_path}/my_gradio/image/美女用户.png",
    "lora_model": f"{abs_path}/lora/output/Qw2.5_7B_lora_meinvyonghu/{find_max_checkpoint(f'{abs_path}/lora/output/Qw2.5_7B_lora_meinvyonghu')}",
    "zh_name":"美女用户"
  },
"guanyu": {"image_path": f"{abs_path}/my_gradio/image/关羽.png",
    "lora_model": f"{abs_path}/lora/output/Qw2.5_7B_lora_guanyu/{find_max_checkpoint(f'{abs_path}/lora/output/Qw2.5_7B_lora_guanyu')}",
    "zh_name":"关羽"
  },
"zhugeliang": {"image_path": f"{abs_path}/my_gradio/image/诸葛亮.png",
    "lora_model": f"{abs_path}/lora/output/Qw2.5_7B_lora_zhugeliang/{find_max_checkpoint(f'{abs_path}/lora/output/Qw2.5_7B_lora_zhugeliang')}",
    "zh_name":"诸葛亮"
  },
"simayi": {"image_path": f"{abs_path}/my_gradio/image/司马懿.png",
    "lora_model": f"{abs_path}/lora/output/Qw2.5_7B_lora_simayi/{find_max_checkpoint(f'{abs_path}/lora/output/Qw2.5_7B_lora_simayi')}",
    "zh_name":"司马懿"
  },
"qun": {"image_path": f"{abs_path}/my_gradio/image/群.png",
    "lora_model": f"{abs_path}/lora/output/Qw2.5_7B_lora_qun/{find_max_checkpoint(f'{abs_path}/lora/output/Qw2.5_7B_lora_qun')}",
    "zh_name":"群"
  },
"zhouyu": {"image_path": f"{abs_path}/my_gradio/image/周瑜.png",
    "lora_model": f"{abs_path}/lora/output/Qw2.5_7B_lora_zhouyu/{find_max_checkpoint(f'{abs_path}/lora/output/Qw2.5_7B_lora_zhouyu')}",
    "zh_name":"周瑜"
  }
}

pass