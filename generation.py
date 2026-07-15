"""
Generation Engine
ACL Mechanistic Equifinality
"""

import os
import torch
import pandas as pd

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from config import *

############################################################
# Load Model
############################################################

print(f"Loading {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    torch_dtype=DTYPE,

    device_map="auto",

    output_hidden_states=True,

    output_attentions=True

)

model.eval()

NUM_LAYERS = model.config.num_hidden_layers

############################################################
# Generate
############################################################

@torch.no_grad()
def generate_reasoning(

        prompt,

        prefix,

        max_tokens=MAX_NEW_TOKENS

):

    full_prompt = prompt + prefix

    inputs = tokenizer(

        full_prompt,

        return_tensors="pt"

    ).to(DEVICE)

    outputs = model.generate(

        **inputs,

        do_sample=False,

        temperature=TEMPERATURE,

        top_p=TOP_P,

        max_new_tokens=max_tokens,

        pad_token_id=tokenizer.eos_token_id

    )

    text = tokenizer.decode(

        outputs[0][inputs.input_ids.shape[1]:],

        skip_special_tokens=True

    )

    return text

############################################################
# Cache Dataset
############################################################

def build_cache(

        dataset,

        prompt_dict,

        cache_file=CACHE_FILE

):

    if os.path.exists(cache_file):

        print("Cache Exists.")

        return pd.read_csv(cache_file)

    records = []

    for sample in tqdm(dataset):

        question = sample["question"]

        gt = sample["answer"]

        chat = tokenizer.apply_chat_template(

            [

                {

                    "role":"user",

                    "content":question

                }

            ],

            tokenize=False,

            add_generation_prompt=True

        )

        item = {

            "question":question,

            "ground_truth":gt,

            "base_prompt":chat

        }

        for name,prefix in prompt_dict.items():

            cot = generate_reasoning(

                chat,

                prefix

            )

            item[name]=cot

        records.append(item)

    df = pd.DataFrame(records)

    df.to_csv(cache_file,index=False)

    return df
