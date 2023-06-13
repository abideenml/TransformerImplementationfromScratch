import torch
import torchtext
# from torchtext.datasets import IWSLT2016
# train_iter, valid_iter, test_iter = IWSLT2016()
# print(torch.__version__)
# print(torchtext.__version__)

from functools import partial

from torch.utils.data import DataLoader

def apply_prefix(task, x):
    return f"{task}: " + x[0], x[1]

from torchtext.datasets import Multi30k

multi_batch_size = 5
language_pair = ("en", "de")
multi_datapipe = Multi30k(split="test", language_pair=language_pair)
task = "translate English to German"

multi_datapipe = multi_datapipe.map(partial(apply_prefix, task))
multi_datapipe = multi_datapipe.batch(multi_batch_size)
multi_datapipe = multi_datapipe.rows2columnar(["english", "german"])
multi_dataloader = DataLoader(multi_datapipe, batch_size=None)

batch = next(iter(multi_dataloader))
input_text = batch["english"]
target = batch["german"]
beam_size = 4

model_input = transform(input_text)
model_output = sequence_generator.generate(model_input, eos_idx=eos_idx, beam_size=beam_size)
output_text = transform.decode(model_output.tolist())

for i in range(multi_batch_size):
    print(f"Example {i+1}:\n")
    print(f"input_text: {input_text[i]}\n")
    print(f"prediction: {output_text[i]}\n")
    print(f"target: {target[i]}\n\n")
