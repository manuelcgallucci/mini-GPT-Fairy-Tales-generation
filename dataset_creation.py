
import os
import pandas as pd

DATA_DIR = "./data-stories"

all_stories = ""
for book_name in os.listdir(DATA_DIR):
    for story_name in os.listdir(os.path.join(DATA_DIR, book_name)):
        df = pd.read_csv(os.path.join(DATA_DIR, book_name, story_name))
        story = ""
        if df["text"].isnull().sum() == 0:
            for text in df["text"]:
                story += text
            
            all_stories += story + "\n"
        else:
            print("Story has a problem:", book_name, story_name)

with open("all_stories.txt", "w") as f:
    f.write(all_stories)