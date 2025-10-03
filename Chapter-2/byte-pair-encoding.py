import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


print("tiktoken version: " , version("tiktoken") )

tokenizer = tiktoken.get_encoding("gpt2")

# text = (
#  "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
#  "of someunknownPlace."
# )
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)

# strings = tokenizer.decode(integers)
# print(strings)


with open("the-verdict.txt", "r", encoding="utf-8") as f:
 raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:       {y}")

for i in range (1, context_size + 1):
 context = enc_sample[:i]
 desired = enc_sample[i]
 print(context, "---->", desired)

for i in range (1, context_size + 1):
 context = enc_sample[:i]
 desired = enc_sample[i]
 print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))



class GPTDatasetV1(Dataset):
 def __init__(self, txt, tokenizer, max_length, stride):
    self.input_ids = []
    self.target_ids = []
    token_ids = tokenizer.encode(txt) 

    for i in range(0, len(token_ids) - max_length, stride): 
        input_chunk = token_ids[i:i + max_length]
        target_chunk = token_ids[i + 1: i + max_length + 1]
        self.input_ids.append(torch.tensor(input_chunk))
        self.target_ids.append(torch.tensor(target_chunk))

 def __len__(self): 
    return len(self.input_ids)
 
 def __getitem__(self, idx): 
    return self.input_ids[idx], self.target_ids[idx]
 
 
 
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
  tokenizer = tiktoken.get_encoding("gpt2")
  dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
  dataloader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        drop_last = drop_last, 
        shuffle = shuffle, 
        num_workers = num_workers
  )

  return dataloader


dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader) 
first_batch = next(data_iter)
second_batch = next(data_iter)
print(first_batch)
print(second_batch)

print("-------------------------------")

dataloader2 = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter2 = iter(dataloader2)
inputs,targets = next(data_iter2)
print("Inputs: \n", inputs) 
print("\nTargets: \n", targets)