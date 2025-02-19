from transformers import AutoModelForCausalLM, AutoTokenizer

## 加载微调后的模型
final_path = './final_saved_path'
model = AutoModelForCausalLM.from_pretrained(final_path)
tokenizer = AutoTokenizer.from_pretrained(final_path)

## 构建推理pipeline
from transformers import pipeline

pipe = pipeline(task:'text-generation',model=model,tokenizer=tokenizer)
prompt = 'tell me some singing skills'
generated_texts = pipe(prompt, max_length=512,num_return_sequences=1)

print('---开始回答---',generated_texts[0]['generated_text'])