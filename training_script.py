import tensorflow as tf
from models.definitions.transformer import *
from utils.constants import *
from utils.optimizer import *
from utils.loss_metrics import *
import argparse
from translation_script import *
from utils.export import *
def train_transformer(training_config):
    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=MODEL_DIMENSION,
        num_heads=NUMBER_OF_HEADS,
        dff=DFF,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        dropout_rate=DROPOUT_PROB)

    print(transformer.summary())
    learning_rate = CustomSchedule(MODEL_DIMENSION)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)
    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])

    transformer.fit(train_batches,
                    epochs=training_config['num_of_epochs'],
                    validation_data=val_batches)
    translator = Translator(tokenizers, transformer)
    translator = ExportTranslator(translator)
    tf.saved_model.save(translator, export_dir='translator')


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    num_warmup_steps = 4000

    #
    # Modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=20)
    # You should adjust this for your particular machine
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=1500)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging", default=True)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=1)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['num_warmup_steps'] = num_warmup_steps

    # Train the original transformer model
    train_transformer(training_config)
    
