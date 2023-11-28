
import os
import pandas as pd
import json

DATA_DIR = "./data-stories"
OUTPUT_DIR = "./stories_clean"

chars = set()
for book_id, book_name in enumerate(os.listdir(DATA_DIR)):
    for story_id, story_name in enumerate(os.listdir(os.path.join(DATA_DIR, book_name))):
        df = pd.read_csv(os.path.join(DATA_DIR, book_name, story_name))
        story = ""
        if df["text"].isnull().sum() == 0:
        
            for text in df["text"]:
                text = text.replace("\n", "")
                story += text.replace("\r", "")

            # print("X:", story.replace("\n", "\n__8__"))                
        else:
            print("Story has a problem:", book_name, story_name)
        
        if story != "":
            with open(os.path.join(OUTPUT_DIR, f"{book_id}_{story_id}.txt"), "w") as f:
                f.write(story)
        
        chars.update(set(story))

meta_dir = {
    "vocab_size": len(chars),
    "chars": sorted(list(chars)),
}
with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
    json.dump(meta_dir, f, indent=2)
