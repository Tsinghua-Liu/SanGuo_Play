import torch
from peft import PeftModel
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from all_characters.dir_path import CHARACTERS  ## 定义选项字典
from api_qwen.guiji_qwen7B import api_qwen7b_generate

class Qwen7BModel:
    def __init__(self, characters_list, model_path, api_mode=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.characters_list = characters_list
        self.model_path = model_path
        self.api_mode = api_mode

        if not self.api_mode:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype="auto", device_map=self.device)

            peft_model_path = list(self.characters_list.values())[0]['lora_model']
            self.peft_model = PeftModel.from_pretrained(self.model, peft_model_path, adapter_name=list(self.characters_list.keys())[0])
            self.peft_model.eval()

            for id, name in enumerate(list(self.characters_list.keys())):
                peft_model_path = list(self.characters_list.values())[id]['lora_model']
                self.peft_model.load_adapter(peft_model_path, adapter_name=name)

    def generate_qwen7B(self, prompt, lora=True):
        if lora:
            with torch.no_grad():
                text = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                generated_ids = self.peft_model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response
        else:
            with self.peft_model.disable_adapter():
                text = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                generated_ids = self.peft_model.generate(
                    **model_inputs,
                    max_new_tokens=1024,

                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response

    def llm_generate_keyword(self, keyword="", api_model=None):
        """
            :keyword  根据游戏规则为词语生成一个非常模糊的描述。
            :return:
            """
        zh_name = "该谜底的一个特点是："
        if not api_model:
            api_model = self.api_mode

        prompt = [
            {
                "role": "system",
                "content": f"""你需要为{keyword}打上一个非常准确的类别标签（例如带电，纸质，不能吃，白色等描述），你只能使用简短的话进行表述，字数少于10个字。描述当中不能与{keyword}有任何重复的字。"""
            }
        ]
        if api_model == True:  # 调用api进行玩游戏
            return f"{zh_name}：{api_qwen7b_generate(prompt)}"

        return f"{zh_name}：{self.generate_qwen7B(prompt, lora=False)}"

    def llm_generate_answer(self, question="", keyword="", api_model=None):
        if not api_model:
            api_model = self.api_mode
        zh_name = "裁判"

        prompt = [
            {
                "role": "system",
                "content": f"""你是一个裁判。
                请你根据谜底来回答人物的问题。谜底：{keyword}。用户的问题是：{question}。
                人物会提出一个与谜底相关的问题。例如谜底是：苹果，人物问题是：“它是红色的吗？” 你需要回答'否'，因为苹果不一定是红色的。人物问题是：“它是否长在树上？” 你需要回答'是'，因为苹果是长在树上的。
                你只能回答'是'或者'否'。"""
            }
        ]
        if api_model:
            return f"{zh_name}：{api_qwen7b_generate(prompt)}"
        return f"{zh_name}：{self.generate_qwen7B(prompt, lora=False)}"

    def llm_generate_judge(self, predict="", keyword="", api_model=None):
        if not api_model:
            api_model = self.api_mode

        zh_name = "裁判"
        prompt = [
            {
                "role": "system",
                "content": f"""你作为裁判，正在主持一个猜谜底的游戏，一些人物参与了游戏。
                人物的猜测由两部分组成：[人物名称：猜测内容]
                如果人物的猜测与谜底一致（可能是一种东西的不同名称），你需要回复'正确'，否则回复'错误'，并且给出一点点提示。
                例如：
                谜底：喜马拉雅山脉。人物的猜测是：（人物姓名：喜马拉雅）。  回复：'正确'。
                谜底：王老吉 。人物的猜测是：（人物姓名：加多宝）。  回复：'错误'，已经差不多了，是其他牌子的饮料。

                下面请你判断用户的猜测，你只能回复'正确'、'错误+提示'（在提示当中禁止提到谜底词）：
                谜底词：{keyword}。人物的猜测是：{predict}。"""
            }
        ]
        if api_model:
            return f"{zh_name}：{api_qwen7b_generate(prompt)}"
        return f"{zh_name}：{self.generate_qwen7B(prompt, lora=False)}"

    def llm_generate_character_question(self, character_name="zhangfei", describe="", history="", api_model=None):
        if not api_model:
            api_model = self.api_mode
        zh_name = self.characters_list[character_name]["zh_name"]


        prompt = [
            {
                "role": "system",
                "content": f"""你需要扮演三国演义中的{zh_name}，你需要以{zh_name}的语言风格进行对话。你和其他古代贤人一起玩一个“猜谜底”的游戏。你的最终目标是猜出一个包含'{describe}'这个词的属性。
                请参考已有的问答[{str(history) if history != "" else f"其中一个特征是：{describe}"}]，然后提出一个可以用“是”或“否”来回答的问题。不要重复已经问过的角度，每个问题都要根据最新的信息来调整方向，目的是缩小答案的范围。"""
            }
        ]
        if api_model:
            return f"{zh_name}：{api_qwen7b_generate(prompt)}"

        self.peft_model.set_adapter(character_name)
        return f"{zh_name}：{self.generate_qwen7B(prompt, lora=True)}"

    def llm_generate_character_predict(self, character_name="zhangfei", describe="", history="", api_model=None):
        if not api_model:
            api_model = self.api_mode

        zh_name = self.characters_list[character_name]["zh_name"]
        prompt = [
            {
                "role": "system",
                "content": f"""你需要扮演三国演义中的{zh_name}，你需要以{zh_name}的语言风格进行对话。你和其他古代贤人一起玩一个“猜谜底”的游戏。
谜底（是一个名词）的属性包含'{describe}'。
                请参考与这个谜底相关的历史问答[{str(history) if history != "" else describe}]，逐步推理，猜测这个谜底应该是什么。
                你只能用一个词语（例如（吹风机，地球...））来猜测谜底到底是什么。
                请你说出一个词语："""
            },
        ]
        if "用户" in zh_name:
            print("打印一下 用户之前的玩家的对话历史 -------", str(history))

        if api_model:
            return f"{zh_name}：{api_qwen7b_generate(prompt)}"

        self.peft_model.set_adapter(character_name)
        return f"{zh_name}：{self.generate_qwen7B(prompt, lora=True)}"
