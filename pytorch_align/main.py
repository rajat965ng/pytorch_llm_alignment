import os, sys
import math

from torch import nn
from tqdm import tqdm
from datetime import datetime
import ipdb
from typing import List, Dict, Union

# pytorch
import torch
import torch.nn
from torch.nn import functional as F

# Import HuggingFace Libraries
import transformers
from datasets import load_dataset, load_from_disk

from pytorch_align.llm import ModelArgs, Llama
from pytorch_align.parameters import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

# optional for debugging entire tensor
torch.set_printoptions(threshold=10000)


# if wandb_log:
#     import wandb
#
#     wandb.init(project=wandb_project, name=wandb_run_name)


# Filter Dataset
# Eliminate entries longer than 512(prompt_max_size). This is important because we want the
# prompt + answer to fit within the total context (1024)
def filter_dataset(examples):
    prompt_length = tokenizer.apply_chat_template(examples['chosen'][:-1], tokenize=True,
                                                  add_generation_prompt=True, return_tensors='pt').size(-1)
    # if add_generation_prompt is true then they are going to add assistant tag at the end of the prompt to
    # encourage the architecture to answer
    if prompt_length < prompt_max_size:  # 512
        return True
    else:
        return False


# Preprocess and tokenize data
def preprocess_dataset(examples: Union[List, Dict]):
    # Take chosen field, eliminate last answer, apply chat template adding assistant prompt
    prompt = [tokenizer.apply_chat_template(item[:-1], tokenize=False, add_generation_prompt=True) for item in
              examples['chosen']]
    chosen = [tokenizer.apply_chat_template(item[:-1], tokenize=False) for item in
              examples['chosen']]
    rejected = [tokenizer.apply_chat_template(item[:-1], tokenize=False) for item in
                examples['rejected']]

    # Let's tokenize
    # HF Tokenizer Dict Format
    # Fields: ids, type_ids, tokens, offsets, attention_mask, special_token_mask, overflowing
    inputs = tokenizer(prompt, max_length=context, padding='max_length', truncation=True, return_tensors='pt')
    # debug: inputs.input_ids[0], inputs.attention_mask[0]

    pos_labels = tokenizer(chosen, max_length=context, padding='max_length', truncation=True, return_tensors='pt')
    # Tokenization of rejected/non-preferred interaction
    neg_labels = tokenizer(rejected, max_length=context, padding='max_length', truncation=True, return_tensors='pt')

    inputs['positive_input_ids'] = pos_labels['input_ids']
    inputs['positive_attention_mask'] = pos_labels['attention_mask']

    inputs['negative_input_ids'] = neg_labels['input_ids']
    inputs['negative_attention_mask'] = neg_labels['attention_mask']

    return inputs


def lr_lambda(current_steps):
    if current_steps <= lr_warmup_steps:
        return float(current_steps) / float(max(1, lr_warmup_steps))
    # Where we are on the path after the warmup until the end of the training
    progress = float(current_steps - lr_warmup_steps) / float(max(1, num_training_steps - lr_warmup_steps))
    # We use it here to control the position in relation to the angle of the calculation with cosine.
    # And the returned value is going to gradually and slowly descend the learning rate from that maximum that
    # we are gonna reach the warm up then slowly down.
    return max(0.0, 0.5 * (1 + math.cos(math.pi * float(0.5) * 2.0 * progress)))


def compute_logps(prompt_attention_mask, chosen_inputs, chosen_attention_mask, logits):
    # compute the PER TOKEN log probabilities of +ve and -ve answers

    # chosen_attention_mask, irrespective of +ve/-ve. We have to get rid of the last one because we're gonna shift
    # the prompt_attention_mask 1 to the left to equalize both of them in sizes.
    # This is going to isolate the answer plus the token before the answer because we have to begin predicting
    # from the token before the answer.
    mask = chosen_attention_mask[:, :-1] - prompt_attention_mask[:, 1:]
    # Get rid of the last position as we have to equalize with the length of the index
    # log_softmax is going apply softmax to transform those predictions into probabilities followed by a log on top
    # look for the probabilities in 2nd dimension, out of 0,1,2 (Three dimensions)
    # to align the index of prediction with token that has to predict -> shift chosen_inputs one to the left
    # put unsqueeze at the end to add few dimensions to compensate other dimensions like batch_size etc. After we finish whole this
    # we'll get rid of them.
    per_token_logps = torch.gather(logits[:, :-1, :].log_softmax(-1), dim=2,
                                   index=(mask * chosen_inputs[:, 1:]).unsqueeze(2)).squeeze(2)

    # We have to isolate probabilities that we care about so
    # we'll multiply per token probabilities times the mask
    return torch.mul(per_token_logps, mask.to(dtype)).sum(dim=1).to(dtype) / mask.sum(dim=1).to(
        dtype)  # active positions in the mask
    pass


if __name__ == "__main__":
    # tokenizing dataset
    # Load the tokenizer in the hugging face format
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    # set our interaction template
    tokenizer.chat_template = "{% for message in messages %}{% if message['role']=='user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

    # Make the padding token equals to the EOS token {which has an ID of 2 in our case} why do we need this? Because
    # different interactions in our datasets will have different length. But when we send this to our
    # architecture/GPU. We would like all interactions to have same length. So we're gonna limit the length of the
    # interactions 512, and the ones that don't reach 512, we're gonna fill the gap that remains, util that length
    # with a padding token. Padding token is going to be the end of sentence token.
    tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        print("Filtering and tokenizing dataset")
        dataset = load_dataset(dataset_name, split="all")
        # now we will tokenize

        # Optional: Filter some of the entries. This dataset is mixture of different alignment data sets and contains
        # some samples that deal with topics like 'hacking' You may want to filter that out. Also, you may want to
        # discard interactions that are too long. Because if they are too long, sequence length may not be able to
        # deal with them. Without filtering 37136 entries vs with filtering 36622 enteries.
        dataset = dataset.filter(lambda r: r['source'] != "toxic-dpo-v0.2")

        # Excluding prompts that are too long
        dataset = dataset.filter(filter_dataset)

        # Preprocess and tokenize dataset
        # if you have issues with multiprocessing change num_proc=1
        # For multiprocessing: num_proc=min(32,os.cpu_count())
        # by default sending batches of 1000
        dataset = dataset.map(preprocess_dataset, batched=True, num_proc=1, remove_columns=dataset.column_names)

        dataset.save_to_disk(dataset_path)

    print(len(dataset))
    print(dataset[0])

    # split the data
    dataset = dataset.shuffle(42).train_test_split(test_size=0.05)
    # features: input_ids, attention_mask
    train_data = dataset['train']
    val_data = dataset['test']

    # structure that efficiently prepare training and validation data for language modeling by combining batching,
    # capabilities, padding optionally
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # setup data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                               collate_fn=data_collator, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator,
                                             num_workers=0)

    # setup our llama architecture
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'base_model.pt'))
    config = checkpoint.pop('config')
    model_args = ModelArgs(
        dim=config.hidden_size,
        n_layers=config.num_hidden_layers,
        n_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        vocab_size=config.vocab_size,
        norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        max_seq_len=context,
        dropout=config.attention_dropout,
        hidden_dim=config.intermediate_size,
        attention_bias=config.attention_bias,
        mlp_bias=config.mlp_bias
    )
    # dim=786, n_layers=12, n_heads=12, vocab=16384, etc.

    model = Llama(model_args)
    model.load_state_dict(checkpoint)
    model = model.to(dtype)
    model = model.to(device)
    model.train()

    if compile:
        print('[INFO] Compiling Model')
        model = torch.compile(model)

    print(sum(p.numel() for p in model.parameters()) / 1e6, " Million parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=10e-8, fused=device == 'cuda',
                                  weight_decay=weight_decay)
    num_training_steps = len(train_loader) * epochs
    print(f"num_training_steps: {num_training_steps}")

    # schedular for lr: first 100 steps, we do the warmup in which we increase linearly the lr.
    # After warmup, we decrease it gradually following a cosine curve.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    # Alignment training loop
    try:
        for e in range(epochs):
            for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True):
                optimizer.zero_grad(set_to_none=True)

                # Move the data to GPU.
                # These are the ids of the preferred interaction with the preferred last answer
                batch['positive_input_ids'] = batch['positive_input_ids'].to(device)
                batch['positive_attention_mask'] = batch['positive_attention_mask'].to(device)
                batch['negative_input_ids'] = batch['negative_input_ids'].to(device)
                batch['negative_attention_mask'] = batch['negative_attention_mask'].to(device)
                # attention mask for the prompt
                batch['attention_mask'] = batch['attention_mask'].to(device)

                # Copy of rejected and preferred interactions
                neg_labels = batch['negative_input_ids'].clone()
                pos_labels = batch['positive_input_ids'].clone()

                # calculating loss. understand what is this extra part of the loss calculation that is gonna help us
                # push the model to favour our preferred answers that are on the dataset versus the rejected ones
                # that are not aligned with our preference. P1: Focus on predict well the next token when it is
                # generating the response mask(all parts of interaction+padding except where the positive answer is)
                # (prompt_attention_mask - 1s at the prompt,
                # 0s everywhere else)
                #                               (batch_attention_mask - 1s in the
                #                                prompt & last preferred answer, 0s everywhere else)
                mask = batch['attention_mask'] * batch['positive_attention_mask']
                # mask will have 1s in prompt and everything will be 0s

                # pos_labels = (IDs(entire Interaction) + positive answer)
                pos_labels = pos_labels * mask.logical_not()  # put 0s in the prompt and preserve last preferred
                # answer, padding tokens have eos (2)

                # location(prompt) = 0s, location(padding_tokens) = 2s. If we minus 100 from both of them and make it
                # large -ve number they will be ignored by Cross Entropy loss calculation.
                pos_labels[pos_labels == 0] = tokenizer.pad_token_id  # eos: 2
                pos_labels[pos_labels == tokenizer.pad_token_id] = -100
                # working with neg_labels to calculate logits. Need them in future for the function compute_logps
                neg_labels[neg_labels == tokenizer.pad_token_id] = -100

                # when model does cross entropy, it is gonna ignore everything except the positions of the
                # positive preferred answer. That is where we really wanna check performance. How well the architecture
                # predicted the next token in that generation of positive preferred answer
                outputs_pos, loss_pos = model(batch['positive_input_ids'], pos_labels)
                outputs_neg, loss_neg = model(batch['negative_input_ids'], neg_labels)  # we'll use outputs_neg for per
                # token log probability calculations of -ve responses that we're gonna use to compare performance (
                # +ve vs -ve responses).

                # calculate per token log probabilities, essential to calculate the ORPO LOG ODDS RATIO

                pos_prob = compute_logps(
                    prompt_attention_mask=batch['attention_mask'],
                    chosen_inputs=batch['positive_input_ids'],
                    chosen_attention_mask=batch['positive_attention_mask'],
                    logits=outputs_pos
                )
                neg_prob = compute_logps(
                    prompt_attention_mask=batch['attention_mask'],
                    chosen_inputs=batch['negative_input_ids'],
                    chosen_attention_mask=batch['negative_attention_mask'],
                    logits=outputs_neg
                )

                # Calculate ORPO Odds ratio
                # using torch.exp for numerical stability
                log_odds = (pos_prob - neg_prob) - (torch.log(1 - torch.exp(pos_prob))) - torch.log(
                    1 - torch.exp(neg_prob))
                sig_ratio = F.sigmoid(log_odds)  # transform the results between 0 & 1
                ratio = torch.log(sig_ratio)  # Exaggerate the sig_ratio

                # Calculate the final loss
                loss = torch.mean(loss_pos - (alpha * ratio).mean()).to(dtype=dtype)

                # Logging
                if i % log_iters == 0:
                    print(
                        f"Epochs [{e}/{epochs}] Step: [{i}/{len(train_loader)}], train loss: {loss.item():.3f}, Odd Ratio: {log_odds.mean().item():.3f}")

                # backpropagate the network to calculate the gradients
                loss.backward()
                # Gradient clipping, in-place we pass the model parameters. max_norm will be grad_clip to prevent
                # gradients explosion by clipping the gradient and maintain stability.
                nn.utils.clip_grad_norm(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                scheduler.step()

            sd = model.state_dict()
            sd['config'] = config
            torch.save(sd, os.path.join(checkpoint_dir, f"{project_name}_{e + 1}.pt"))


    except KeyboardInterrupt:
        print("Training interrupted. Cleaning up .....")
    finally:
        # Release GPU Memory
        torch.cuda.empty_cache()
        print("GPU memory released")
