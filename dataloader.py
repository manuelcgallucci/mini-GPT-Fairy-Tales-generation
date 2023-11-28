
from torch.utils.data import Dataset
import numpy as np
import os
import json
import torch

OUTPUT_DIR = "./stories_clean"
META_DIR = os.path.join(OUTPUT_DIR, "meta.json")

class DataLoaderStory():
    def __init__(self, dataset, block_size, batch_size):
        self.dataset = dataset
        self.block_size = block_size
        self.batch_size = batch_size

    def get_batch(self):
        # data_id = np.random.choice(len(self.dataset), 1, p=self.dataset.data_weights)
        data_id = np.random.choice(len(self.dataset), 1)[0]
        data = self.dataset[data_id]
        ix = torch.randint(len(data) - self.block_size, (self.batch_size, ))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y

class DatasetStories(Dataset):
    def __init__(self, stories_dir, device, seed=np.random.randint(0, 100000)):
        """stories_dir: A list of stories paths"""
        
        with open(META_DIR, "r") as f:
            meta = json.load(f)            
        str2int = {ch:i for i, ch in enumerate(meta["chars"])}
        encode = lambda s: [str2int[c] for c in s]

        self.data_list = []
        self.data_weights = []
        for st_d in stories_dir:
            with open(st_d, "r") as f:
                story = f.read()
            self.data_list.append(torch.Tensor(encode(story)).to(torch.int64).to(device))
            self.data_weights.append(len(self.data_list[-1]))
        self.data_weights = [x / sum(self.data_weights) for x in self.data_weights] # not used 
        
        # print(self.data_weights)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]

def get_train_val_dirs(stories_dir, train_ptg=0.8):
    dirs = [os.path.join(OUTPUT_DIR, x) for x in os.listdir(stories_dir) if x[-4:] == ".txt"]
    np.random.shuffle(dirs)
    # for st_d in dirs:
    #     with open(st_d, "r") as f:
    #         story = f.read()
    #     len(story)
    idx = int(len(dirs) * train_ptg)
    train_dirs = dirs[idx:]
    val_dirs = dirs[:idx]

    return train_dirs, val_dirs

# if __name__ == "__main__":
#     stories_dir = [os.path.join(OUTPUT_DIR, x) for x in os.listdir(OUTPUT_DIR) if x[-4:] == ".txt"]
#     DatasetStories(stories_dir, device="cpu")