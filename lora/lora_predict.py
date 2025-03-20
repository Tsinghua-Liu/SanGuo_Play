import torch
from peft import PeftModel
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

model_path = "---"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype="auto",device_map = device)

# 加载角色
characters_list = {}
with open("../characters.json", "r", encoding="utf-8") as f:
    characters_list = json.load(f)

#创建 peft model 的个数要根据已经选择的人物进行选择
peft_model_path =  peft_model_path = r"output/Qwen2.5_instruct_lora_"+ list(characters_list.keys())[0] +"/checkpoint-360"
peft_model = PeftModel.from_pretrained(model,peft_model_path, adapter_name=list(characters_list.keys())[0])
for name in list(characters_list.keys()):
    peft_model_path = r"output/Qwen2.5_instruct_lora_"+name +"/checkpoint-360"
    #peft_model_list.append(PeftModel.from_pretrained(model,peft_model_path, adapter_name=name))
    peft_model.load_adapter(peft_model_path, adapter_name = name)


def character_llm_generate_speak(messages,character_name = "caocao" ,secret = ""):
    """
    :param messages:  专属词汇；所有人的多轮历史消息；投票记录
    :param character_name:
    :return:
    """

    peft_model.set_adapter(character_name)

    zh_name = characters_list[character_name]["zh_name"]
    prompt = [
        {
            "role": "system",
            "content": f"""你需要扮演三国演义中的{zh_name}，假设你穿越到了古代，你和其他古代角色一起参与《谁是卧底》游戏。
            《谁是卧底》是一款语言推理游戏，平民持有相同词（如'电梯'），卧底持有相似易混词（如'扶梯'），每轮玩家需用一句话描述自己的词汇（如'垂直运输工具'）但不能直接说出词语，通过分析他人描述的合理性投票淘汰疑似卧底者，平民需在卧底存活人数降至与平民相等前找出所有卧底，而卧底则要伪装成平民混淆视听直至存活到最后。"""
        },
        {
            "role": "user",
            "content": f"""请作为{zh_name}参与谁是卧底游戏，你的专属词汇'{secret}'、其他玩家的历史发言[{messages['history']}]及投票记录[{messages['vote']}]，生成一句描述。禁止与历史记录中的发言相似，你的发言必须从新的角度来描述'{secret}'。"""
        }
    ]
    with torch.no_grad():
        text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = peft_model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


def character_llm_generate_vote(messages,character_name = "zhangfei",secret = "",character_list = []):
    """
    :param messages:  专属词汇；所有人的多轮历史消息；投票记录
    :param character_name:
    :return:
    """

    peft_model.set_adapter(character_name)

    zh_name = characters_list[character_name]["zh_name"]
    prompt = [
        {
            "role": "system",
            "content": f"""你需要扮演三国演义中的{zh_name}，假设你穿越到了古代，你和其他古代角色一起参与《谁是卧底》游戏。
            《谁是卧底》是一款语言推理游戏，人数较多的平民持有相同词（如'电梯'），人数较少的卧底持有相似易混词（如'扶梯'），每轮玩家会使用一句话描述自己的词汇（如'垂直运输工具'），通过分析他人描述的合理性投票淘汰疑似卧底者，平民需要找出谁是卧底，而卧底则要伪装成平民混淆视听直至存活到最后。"""
        },
        {
            "role": "user",
            "content": f"""请作为{zh_name}参与谁是卧底游戏，根据以下信息，投出最可疑的玩家：你的专属词汇'{secret}'、所有人物的历史发言[{messages['history']}]、历史投票记录[{messages['vote']}]、存活的玩家列表{[character_list]}。
            如果你认为你是平民，优先选择与多数玩家描述方向明显偏离的玩家。如果你认为你是卧底，你需要隐藏好自己。你只能从存活的玩家列表中选择一个玩家进行投票或者弃票。
            请你直接输出玩家的姓名，例如'宋慈' ，或者输出 '弃票'。"""
        }
    ]
    with torch.no_grad():
        text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = peft_model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response



def test_():
    messages = {}
    messages["secret"] = "冰箱"
    messages["history"] = """第1轮：

    第1轮
    玩家A（刘备）："此物乃居家必备，户户皆需。"
    张飞："可藏食保鲜，效果甚佳。"
    玩家C（关羽）："常置于庖厨，其形甚巨。"
    玩家D（曹操）："吾家此物，常作嗡嗡之声。"
    玩家E（诸葛亮）："可冻冰酪，夏日不可或缺。"
    玩家F（赵云）："需时常除霜，稍显繁琐。"

    第2轮
    玩家A（刘备）："其功甚大，耗电亦多。"
    张飞："分层而设，可纳万物。"
    玩家C（关羽）："门有把手，启闭甚便。"
    玩家D（曹操）："压缩机作，噪声不绝。"
    玩家E（诸葛亮）："冷冻之处，冰霜可结。"
    """
    messages["vote"] = """
        第1轮：赵云被投出（3票）
        """

    response = character_llm_generate_vote(messages=messages, character_name="zhangfei",character_list = ['刘备','关羽','曹操','诸葛亮'])
    print(response)


if __name__=='__main__':
    test_()
    pass
