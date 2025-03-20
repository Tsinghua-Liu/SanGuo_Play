import torch
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import argparse

def main(character_name, num_train_epochs, model_path, resume_from_checkpoint):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载数据集
    ds = Dataset.load_from_disk(f"train_data/sanguo/{character_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map=device)

    # 加载现有LoRA适配器或初始化新适配器
    if resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)
        print(f"Resumed training from checkpoint: {resume_from_checkpoint}")
    else:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj"],
            modules_to_save=["word_embeddings"],
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = get_peft_model(model, config)
        print("Initialized new LoRA adapter")
    model.print_trainable_parameters()

    # 数据处理函数（保持不变）
    def process_func(example):
        MAX_LENGTH = 512

        # 完整对话拼接
        formatted_str = "".join(
            f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            for msg in example['messages']
        )

        # 分词处理（保留完整对话）
        tokenized = tokenizer(
            formatted_str,
            max_length=MAX_LENGTH,
            truncation=True,
        )

        # 标签生成逻辑
        labels = []
        for msg in example['messages']:
            # 对每条消息单独分词
            msg_str = f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            msg_tokens = tokenizer(msg_str, add_special_tokens=False).input_ids
            if msg['role'] == 'assistant':
                # Assistant部分作为训练目标
                labels.extend(msg_tokens)
            else:
                # 其他角色部分mask掉（不参与loss计算）
                labels.extend([-100] * len(msg_tokens))

        input_ids = tokenized["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = tokenized["attention_mask"] + [1]
        labels = labels + [tokenizer.pad_token_id]

        # 截断与填充处理
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        if len(input_ids) != len(labels):
            print("错误: len(input_ids) != len(labels)")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

    # 配置训练参数（关键修改：save_total_limit=None）
    args = TrainingArguments(
        output_dir=f"./output/Qw2.5_7B_lora_{character_name}",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        logging_steps=10,
        num_train_epochs=num_train_epochs,
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        warmup_steps=10,
        weight_decay=0.1,
        save_strategy="epoch",
        save_total_limit=None,  # 禁用检查点数量限制
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    # 控制训练恢复方式
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--character_name', type=str, default="caocao")
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--model_path', type=str, default="/path/to/default/model")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    args = parser.parse_args()
    main(args.character_name, args.num_train_epochs, args.model_path, args.resume_from_checkpoint)