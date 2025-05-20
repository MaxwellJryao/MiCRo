from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import torch
import os

model_name = 'meta-llama/Llama-3.2-3B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval().cuda()


ds = load_dataset('microsoft/rpr')['train']
hidden_states = []

with torch.no_grad():
    for item in tqdm(ds):
        criteria = item['criteria_x']
        tokens = tokenizer(criteria, return_tensors='pt')['input_ids'].cuda()
        outputs = model(tokens)
        hidden_state = outputs[0][0,-1].cpu() # last hidden state, last token
        hidden_states.append(hidden_state)

        criteria = item['criteria_y']
        tokens = tokenizer(criteria, return_tensors='pt')['input_ids'].cuda()
        outputs = model(tokens)
        hidden_state = outputs[0][0,-1].cpu() # last hidden state, last token
        hidden_states.append(hidden_state)

hidden_states = torch.stack(hidden_states)
torch.save(hidden_states, 'hidden_states.pt')
print('done')
