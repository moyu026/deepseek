## 第一步，测试微调之前的模型是否可用
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from data_prepare import samples

## 加载微调之前的模型，deepseek-r1-1.5b
model_name = ''  # deepseek-r1-1.5b模型文件的地址
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
#
# print("------模型加载成功-------")

## 第二步，制作数据集
with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        jsom_line = json.dump(s, ensure_ascii=False)
        f.write(jsom_line + '\n')
    else:
        print("prepare data finished")

## 第三步，准备训练集和测试集
from datasets import load_dataset

dataset = load_dataset(path:"json", data_files = {"train:datasets.jsonl"}, split = "train")
print("数据数量：", len(dataset))

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"train dataset len: {len(train_dataset)}")
print(f"test dataset len: {len(eval_dataset)}")

print("-----完成训练数据的准备工作-------")


## 第四步：编写tokenizer处理工具
def tokenizer_function(many_samples):
    texts = [f"{prompt}\n{completion}" for prompt, completion in
             zip(many_samples['prompt'], many_samples['completion'])]
    tokens = tokenizer(texts, truncation=True, max_length=512, padding='max_length')
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

tokenized_train_dataset = train_dataset.map(tokenizer_function, batch=True)
tokenized_eval_dataset = eval_dataset.map(tokenizer_function, batch=True)

print("-----完成tokenize------")
# print(tokenized_train_dataset[0])

## 第五步，量化设置
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretained(model_name,quantization_config=quantization_config,device_map='auto')
print("已经完成量化模型加载")

## 第六步，Lora微调设置
from peft import get_peft_model,LoraConfig,TaskType
lora_config = LoraConfig(
    r=8,lora_alpha=16,lora_dropout=0.05,task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model,lora_config)
model.print_trainable_parameters()
print("----lora微调设置完毕-----")

## 第七步，设置训练参数并训练
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./finetune_models',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=10,
    learning_rate=3e-5,
    logging_dir='./logs',
    run_name='deepseek-r1-distill-finetune'
)

## 定义训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

## 开始训练
trainer.train()

## 第八步，保存Lora模型
save_path = './saved_models'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

## 保存全量模型
final_save_path = './final_saved_path'

from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretained(model_name)
model = PeftModel.from_pretrained(base_model, save_path)
model = model.merge_and_unload()

model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)










