import torch
from bigram_v6_scaleanddropout import BigramLanguageModel

# Inference length
story_length = 2000

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BigramLanguageModel()
m = model.to(device)

with open("all_stories.txt", "r", encoding="utf-8") as f:
    dataset = f.read()

chars = sorted(list(set(dataset)))
vocab_size = len(chars)
# Mapping from vocab chars to ints 
# Very small vocal size tokenizer, gpt-2 uses 50k 
int2str = {i:ch for i, ch in enumerate(chars)}
decode = lambda l: "".join([int2str[i] for i in l])

context = torch.zeros((1, 1), device=device, dtype=torch.long)
with open("output.txt", "w") as f:
    f.write(decode(m.generate(context, max_new_tokens=story_length)[0].tolist()))
