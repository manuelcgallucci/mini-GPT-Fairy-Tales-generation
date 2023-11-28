import torch
import json

from bigram_model import *

OUTPUT_DIR = "./stories_clean"
META_DIR = os.path.join(OUTPUT_DIR, "meta.json")

# Inference length
story_length = 2000

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BigramLanguageModel()
model.to(device)

model.load_state_dict(torch.load("model.pt"))
model.eval()

with open(META_DIR, "r") as f:
    meta = json.load(f)   

chars = meta["chars"]
vocab_size = meta["vocab_size"]
# Mapping from vocab chars to ints 
# Very small vocal size tokenizer, gpt-2 uses 50k 
int2str = {i:ch for i, ch in enumerate(chars)}
decode = lambda l: "".join([int2str[i] for i in l])

context = torch.zeros((1, 1), device=device, dtype=torch.long)
with open("output.txt", "w") as f:
    f.write(decode(model.generate(context, max_new_tokens=story_length)[0].tolist()))
