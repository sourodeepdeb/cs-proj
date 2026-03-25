from google.colab import drive
drive.mount('/content/drive')
import os
import pandas as pd
from openai import OpenAI

# connect to google drive
drive.mount("/content/drive")

# set working directory path
project_path = "---"
os.chdir(project_path)

# initialize openai api client
client = OpenAI(api_key="---")

# load the reddit dataset
df = pd.read_csv("sarcasticComments.csv")

# filter for sarcastic labels
sarcastic_df = df[df["label"] == 1].copy()

# take first thousand rows
sarcastic_df = sarcastic_df.head(1000)

def create_embedding(text):
    # remove new line characters
    text = str(text).replace("\n", " ")
    # generate numeric vector representation
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

# apply embeddings to comments
sarcastic_df["parent_embedding"] = sarcastic_df["parent_comment"].apply(create_embedding)

# save results to csv
sarcastic_df.to_csv("sarcasmEmbeddings.csv", index=False)

# how you know your done
print("done")
