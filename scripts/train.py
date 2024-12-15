from load_data import decoder_input_data, decoder_target_data, encoder_input_data
import os

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from load_data import (
    decoder_input_data,
    decoder_target_data,
    encoder_input_data,
    max_encoder_seq_length,
    max_decoder_seq_length,
    num_encoder_tokens,
    num_decoder_tokens,
    reverse_target_char_index,
    target_token_index,
)
from models import DeepEnsemble

batch_size = 32
epochs = 20

no_models = 3
threshold = int(round(no_models * 2 / 3) / no_models * 100) / 100

de = DeepEnsemble(
    no_models=no_models,
    threshold=threshold,
    max_encoder_seq_length=max_encoder_seq_length,
    max_decoder_seq_length=max_decoder_seq_length,
    num_encoder_tokens=num_encoder_tokens,
    num_decoder_tokens=num_decoder_tokens,
    reverse_target_char_index=reverse_target_char_index,
    target_token_index=target_token_index,
)

# plot_model(de.models[0][0], show_shapes=True, to_file="images/model.png")
# plot_model(de.models[0][1], show_shapes=True, to_file="images/encoder.png")
# plot_model(de.models[0][2], show_shapes=True, to_file="images/decoder.png")

history = de.fit(
    x=[encoder_input_data, decoder_input_data],
    y=decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)

save_path = "models/DE_ENG_v2"

if not os.path.exists(save_path):
    de.save(save_path)
else:
    print("Already exists.")
