from datasets import Dataset
from transformers import AutoTokenizer
import json
import os

# 使用datasets 处理数据，转化为可以用来训练角色模型的数据格式

# 文件来源路径、保存路径
save_file_path = "allname"

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 根据项目结构回退到根目录（假设脚本在 sanguo_peft/sanguo/... 下）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))  # 根据实际层级调整
# 构建完整文件路径
file_path = os.path.join(
    project_root,
    f"datasets/raw_data/sanguo_dataprocess/train_data/train_data_{save_file_path}.json"
)

with open(file_path,"r",encoding='utf-8') as f:
    datastes = json.load(f)

# 将数据转换为 Dataset 格式
def convert_to_dataset(datastes):
    # 将每条数据转换为字典格式
    formatted_data = [{"messages": messages} for messages in datastes]
    # 创建 Dataset

    ds = Dataset.from_list(formatted_data)
    return ds

# 创建 Dataset
ds = convert_to_dataset(datastes)
# 保存 Dataset 到磁盘
ds.save_to_disk(save_file_path)
# 加载 Dataset
ds = Dataset.load_from_disk(save_file_path)

# 数据处理函数
def process_func(example):
    # 初始化拼接字符串
    formatted_str = ""

    # 遍历每个对话片段
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        # 拼接对话格式
        formatted_str += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    return {"text": formatted_str}

# 使用 map 方法处理数据
tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
# 打印处理后的数据
print(tokenized_ds[0])