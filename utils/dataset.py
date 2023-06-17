import logging
import time
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text
from .constants import BATCH_SIZE, MAX_TOKENS, BUFFER_SIZE


def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))

def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    return (pt, en_inputs), en_labels


if __name__ == "__main__":
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train_examples, val_examples, test_examples = examples['train'], examples['validation'], examples['test']
    model_name = 'ted_hrlr_translate_pt_en_converter'
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True
    )
    tokenizers = tf.saved_model.load(model_name)
    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    # Check vocab size
    print(f'Source vocabulary size={tokenizers.pt.get_vocab_size()}')
    print(f'Target vocabulary size={tokenizers.en.get_vocab_size()}')
