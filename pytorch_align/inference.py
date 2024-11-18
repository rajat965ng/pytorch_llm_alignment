import transformers
from torch import torch
import os

from pytorch_align.llm import ModelArgs, Llama

use_orpo = True  # use aligned checkpoint or not
num_answers = 3
temp = 1
topk = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_path = "tokenizers/tok16384"
model_path = "./models/"

if __name__ == "__main__":

    if use_orpo == True:
        model_inf, context = "aligned_model.pt", 1024  # ORPO is trained with context of 1024
        print("Mode::Using Orpo aligned model")
    else:
        model_inf, context = "base_model.pt", 512  # The original was trained with context of 512
        print("Mode::Using pretrained model without alignment")

    print(f"Using model {model_inf}")

    # Load model and extract config
    checkpoint = torch.load(os.path.join(model_path, model_inf), map_location=device)
    config = checkpoint.pop("config")

    # temporary fix if the model was trained and saved with torch.compile
    # The _orig_mod. prefix in your model's state dictionary keys is related to
    # how PyTorch handles compiled models, specifically when using the torch.compile function
    # When torch.compile is used, PyTorch might wrap the original model in a way that modifies
    # the names of its parameters and buffers. This wrapping can prepend a prefix like _orig_mod.
    # We remove those wrappings to make the checkpoint compatible with the non compiled version of the model
    new_dict = dict()
    for k in checkpoint.keys():
        if k.startswith("_orig_mod."):
            # print("Removing _orig_mod wrapping")
            new_dict[k.replace("_orig_mod.", "")] = checkpoint[k]
        else:
            new_dict[k] = checkpoint[k]

    # Setup tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

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

    # Instantiate model, load params, move to device
    model = Llama(model_args)
    model.load_state_dict(new_dict)
    if device.type == 'cuda':
        model = model.to(torch.bfloat16)
        model = model.to(device)
    model.eval()

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size / 1e6:.2f} M parameters")

    # Interactive loop
    while True:
        qs = input("Enter text (q to quit) >>> ")
        if qs == "":
            continue
        if qs == 'q':
            break

        # we activate chat template only for ORPO model because it was trained with it
        if use_orpo:
            qs = f"<s> <|user|>\n{qs}</s>\n<s> <|assistant|> "

        x = tokenizer.encode(qs)
        x = torch.tensor(x, dtype=torch.long, device=device)[None, ...]

        for ans in range(num_answers):
            with torch.no_grad():
                y = model.generate(
                    x,
                    max_new_tokens=256,
                    temperature=temp,
                    top_k=topk
                )

            response = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)

            output = model.clean_response(response)

            print("################## \n")
            print(f"### Answer {ans + 1}: {output}")