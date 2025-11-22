from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformer_lens import HookedTransformer

def load_gpt2():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

def generate_text(prompt, max_length=50):
    model, tokenizer = load_gpt2()
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=max_length)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text

def run_activation_patching(prompt):
    model = HookedTransformer.from_pretrained("gpt2-small")
    tokens = model.to_tokens(prompt)
    
    activations = {}
    def hook_fn_hook(value, hook):
        activations[hook.name] = value.detach().cpu().numpy()
    
    hooks = []
    for i in range(model.cfg.n_layers):
        hooks.append(model.hook.add_hook(f"blocks.{i}.mlp.hook_post", hook_fn_hook))
    
    _ = model(tokens)
    
    for h in hooks:
        h.remove()
    
    return activations
