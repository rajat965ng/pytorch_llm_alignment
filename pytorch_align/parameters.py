# training parameters
import datetime

import torch

batch_size = 1
epochs = 1  # beyond 3, the underline pre-trainined model start to overfit
lr = 6e-5
lr_warmup_steps = 100  # for first 100 iterations lr will be low then there will be a spike in lr.
context = 1024  # The base model has been trained on SL of 512 + we need to combine the SL (512) of our dataset as well
alpha = 0.5  # weighting/scaling fractor for ORPO(technique of alignment) odd ratio.
# in case of alignment we will not only going to calculate loss but also odds ratio.
prompt_max_size = 512  # Limit for the prompt part of the interaction, prompt+response = 1024 tokens.
# As context = 1024, we don't want to go beyond that.
compile = False  # way of improving the performance of pytorch calculations. If you set it to true, you'll need less
# memory, it will  compute faster, but it's not compatible with every system.
dtype = torch.bfloat16
log_iters = 50  # for every how many iterations you want to log the statistics to weights

# hyperparameters of architecture
dropout = 0
grad_clip = 1.0  # way of clipping the magnitude of gradients to prevent them from exploding and becoming unstable.
weight_decay = 0.0  # regularization mechanism, that is gonna limit the size, the magnitude of the weights,
# to encourage the network to be more flexible

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# logging
project_name = "aligntest"
wandb_log = True
wandb_project = project_name
wandb_run_name = "aligntest-run" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# this is where the tokenized data be stored
dataset_path = "./data/orpo_dataset"
dataset_name = "mlabonne/orpo-dpo-mix-40k"
tokenizer_path = "tokenizers/tok16384"
checkpoint_dir = "./models/"
