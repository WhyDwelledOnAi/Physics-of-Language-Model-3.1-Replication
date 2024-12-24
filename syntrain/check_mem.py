import json, random
from transformers import LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

device = 'cpu'

def load_model(path):
    tokenizer = LlamaTokenizer.from_pretrained("tokenizer.model")
    model = LlamaForCausalLM.from_pretrained(path).to(device)
    return tokenizer, model

def load_bio_data(path):
    with open(path, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f.readlines()]
    data = random.choices(data, k=20)
    return data

def load_qa_data(path):
    qa_data = pd.read_parquet(path)['content'].tolist()
    qa_data = random.choices(qa_data, k=40)
    data = []
    for i in range(len(qa_data)):
        text = qa_data[i]
        prefix = text[:text.find('\nAnswer:')+8]
        answer = text[text.find('\nAnswer:')+8:]
        data.append({'prefix': prefix, 'answer': answer})
    return data

def check_memory(tokenizer, model, data, path):
    # with open(path, 'w', encoding='utf8') as f:
    #     f.write('')
    for info in tqdm(data):
        prefix = info['prefix']
        inputs = tokenizer(prefix, return_tensors="pt").to(device)
        generate_ids = model.generate(input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask, max_length=128,
            num_beams=4)
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=False)[0]
        info['response'] = response
        with open(path, 'a+', encoding='utf8') as f:
            f.write(json.dumps(info, ensure_ascii=False) + '\n')

def check_infering(tokenizer, model, data, path):
    is_right = 0
    for info in tqdm(data):
        prefix = info['prefix']
        inputs = tokenizer(prefix, return_tensors="pt").to(device)
        generate_ids = model.generate(input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask, max_length=128)
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=False)[0]
        info['response'] = response
        if info['answer'] in response:
            is_right += 1
        with open(path, 'a+', encoding='utf8') as f:
            f.write(json.dumps(info, ensure_ascii=False) + '\n')
    print(is_right, len(data))


def check_ppl(tokenizer, model, data, path):
    for info in tqdm(data):
        prefix = info['prefix']
        answer = info['answer']

        inputs = tokenizer(prefix+answer, return_tensors="pt").to(device)
        outputs = model(input_ids = inputs.input_ids,
                        attention_mask = inputs.attention_mask, 
                        output_hidden_states=False, 
                        output_attentions=False)
        answer_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids.squeeze())[1:]
        response_tokens = tokenizer.convert_ids_to_tokens(outputs.logits.argmax(-1).squeeze())[:-1]
        print(tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=False))
        for i in range(len(answer_tokens)):
            print(answer_tokens[i], response_tokens[i])
        break


if __name__ == "__main__":
    epoch = 99
    tokenizer, model = load_model(f'output/tiny/Epoch{epoch}')
    # data = load_bio_data('data/tiny_balanced/check_data.jsonl')
    data = load_qa_data('data/tiny_balanced/train_qas.parquet')
    check_memory(tokenizer, model, data, f'output/tiny/Epoch{epoch}/check_mem.jsonl')
    # check_ppl(tokenizer, model, data, f'output/tiny/Epoch{epoch}/check_ppl.jsonl')