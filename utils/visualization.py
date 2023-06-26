import tensorflow as tf
import matplotlib.pyplot as plt 
from utils.dataset import *
from translation_script import *
import argparse

def plot_attention_head(in_tokens, translated_tokens, attention):
  # The model didn't generate `<START>` in the output. Skip it.
  translated_tokens = translated_tokens[1:]

  ax = plt.gca()
  ax.matshow(attention)
  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(translated_tokens)))

  labels = [label.decode('utf-8') for label in in_tokens.numpy()]
  ax.set_xticklabels(
      labels, rotation=90)

  labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
  ax.set_yticklabels(labels)


def plot_attention_weights(sentence, translated_tokens, attention_heads):
  in_tokens = tf.convert_to_tensor([sentence])
  in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
  in_tokens = tokenizers.pt.lookup(in_tokens)[0]

  fig = plt.figure(figsize=(16, 8))

  for h, head in enumerate(attention_heads):
    ax = fig.add_subplot(2, 4, h+1)

    plot_attention_head(in_tokens, translated_tokens, head)

    ax.set_xlabel(f'Head {h+1}')

  plt.tight_layout()
  plt.show()


def plot_graphs(translation_config):
   translator = tf.saved_model.load('translator')
   translated_text, translated_tokens, attention_weights = translator(
    tf.constant(translation_config['source_sentence']))
   print_translation(translation_config['source_sentence'], translated_text)
   plot_attention_weights(translation_config['source_sentence'], translated_tokens, attention_weights[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_sentence", type=str, help="source sentence to translate into target", default="How are you doing today?")
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    translation_config = dict()
    for arg in vars(args):
        translation_config[arg] = getattr(args, arg)

    # Translate the given source sentence
    plot_graphs(translation_config)
