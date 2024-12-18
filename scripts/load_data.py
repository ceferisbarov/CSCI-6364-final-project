import json
import pandas as pd
import numpy as np

"""
This script takes the processed data, and prepares it for downstream tasks.
"""

# Load the data
data = pd.read_csv("data/english.csv")
data = data.sample(frac=1, random_state=42)
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# data = pd.read_csv("data/processed.csv")
# data = data.sample(frac=1, random_state=42)
# train_data = data.sample(frac=0.8, random_state=42)
# test_data = data.drop(train_data.index)

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

for idx, row in train_data.iterrows():
    input_text = row[0]
    target_text = row[1]
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
input_characters.insert(0, " ")
target_characters = sorted(list(target_characters))
target_characters.insert(0, " ")

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

with open("data/input_token_index.json", "w") as fp:
    json.dump(input_token_index, fp)

with open("data/target_token_index.json", "w") as fp:
    json.dump(target_token_index, fp)
