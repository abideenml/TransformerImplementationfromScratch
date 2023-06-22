import tensorflow as tf
from utils.constants import *
from utils.dataset import *
from embeddings import *
from attention import *
from feedforward import *
from encoder import *
from decoder import *


  
class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits
  

# Testing the correctness of the transformer model - feel free to ignore - I used it during model development
if __name__ == "__main__":
    for (pt, en), en_labels in train_batches.take(1):
        break
    embed_pt = PositionalEmbedding(vocab_size=tokenizers.pt.get_vocab_size(), d_model=512)
    embed_en = PositionalEmbedding(vocab_size=tokenizers.en.get_vocab_size(), d_model=512)

    pt_emb = embed_pt(pt)
    en_emb = embed_en(en)
    
    transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=MODEL_DIMENSION,
    num_heads=NUMBER_OF_HEADS,
    dff=DFF,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=DROPOUT_PROB)

    output = transformer((pt, en))
    print(en.shape)
    print(pt.shape)
    print(output.shape)
    attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
    print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)