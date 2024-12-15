import time
import pandas as pd
from sklearn.metrics import accuracy_score

from load_data import (
    reverse_target_char_index,
    test_data,
    train_data,
    max_encoder_seq_length,
    max_decoder_seq_length,
    num_encoder_tokens,
    num_decoder_tokens,
    target_token_index,
    input_token_index,
)
from models import DeepEnsemble
from utils import plot_results, CER, WER
from tqdm import tqdm

chars = reverse_target_char_index.values()
test_data.dropna(inplace=True)
# The line below is used to use a fraction of the test dataset, mostly during debugging
# Set the frac argument to 1 to retrieve the complete dataset
# Or simply comment out the line
# test_data = test_data.sample(frac=0.001)
test_data = test_data[test_data["text"].apply(lambda s: all(c in chars for c in s))]
test_data = test_data[test_data["text"].str.len() <= train_data["text"].str.len().max()]


from itertools import islice

import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=5)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, 0, 1)

def predict(input_word):
    suggestions = sym_spell.lookup(input_word, Verbosity.CLOSEST, max_edit_distance=5, include_unknown=True)

    for suggestion in suggestions:
        return suggestion.term
    

output = []
start = time.time()
for i in tqdm(range(len(test_data["text"]))):
    pred = predict(
        test_data["text"].iloc[i]
    )
    output.append(pred.strip(" \n\r\t"))

end = time.time()
duration = end - start
latency = round(duration / len(test_data), 3)

test_data["prediction"] = output

wer = round(WER(y_true=test_data.label, y_pred=output) * 100, 2)
cer = round(CER(y_true=test_data.label, y_pred=output) * 100, 2)
acc = f"WER={wer} CER={cer}"

no_models = 0
treshold=0

plot_results(
    data=test_data,
    method="ensemble",
    accuracy=acc,
    latency=latency,
    no_models=no_models,
    treshold=treshold,
)
