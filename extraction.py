"""
Representation Extraction
"""

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os

from generation import *

def attention_rollout(attentions):

    seq_len = attentions[0].shape[-1]

    rollout = np.eye(seq_len)

    all_rollout = []

    for layer in attentions:

        A = layer[0].mean(dim=0).cpu().numpy()

        A += np.eye(seq_len)

        A /= (

            A.sum(

                axis=-1,

                keepdims=True

            )

            +1e-12

        )

        rollout = A @ rollout

        all_rollout.append(

            rollout.copy()

        )

    return all_rollout

@torch.no_grad()
def forward(
    full_text,
    return_inputs=True,
    return_outputs=True,
):

    inputs = tokenizer(
        full_text,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=True,
        use_cache=False,
        return_dict=True
    )

    if return_inputs and return_outputs:
        return inputs, outputs

    elif return_outputs:
        return outputs

    return inputs

def extract_hidden(outputs):

    return np.stack(

        [

            h[0].cpu().numpy()

            for h in outputs.hidden_states

        ],

        axis=0

    )


def extract_last_hidden(outputs):

    hs=[]

    for h in outputs.hidden_states:

        hs.append(

            h[0,-1].cpu().numpy()

        )

    return hs

def extract_token_hidden(outputs):

    result=[]

    for h in outputs.hidden_states:

        result.append(

            h[0].cpu().numpy()

        )

    return result


def extract_logits(outputs):

    return outputs.logits.cpu().numpy()


def extract_probability(outputs):

    return F.softmax(

        outputs.logits[0,-1],

        dim=-1

    ).cpu().numpy()


def extract_entropy(outputs):

    probs = F.softmax(

        outputs.logits,

        dim=-1

    )

    entropy = (

        -(probs * torch.log(probs+1e-12)).sum(dim=-1)

    )

    return entropy.cpu().numpy()


def extract_attention(outputs):

    return [

        x.cpu().numpy()

        for x in outputs.attentions

    ]
    
def extract_rollout(outputs):

    return attention_rollout(

        outputs.attentions

    )


def extract_ids(inputs):

    return inputs.input_ids[0].cpu().numpy()


def extract_tokens(inputs):

    ids = inputs.input_ids[0]

    return tokenizer.convert_ids_to_tokens(

        ids

    )

def extract_logit_lens(outputs):

    lm_head = model.lm_head

    vocab=[]

    probs=[]

    ids=[]

    for h in outputs.hidden_states:

        logits = lm_head(

            h[0,-1]

        )

        p = F.softmax(

            logits,

            dim=-1

        )

        idx = torch.argmax(p).item()

        token = tokenizer.decode([idx])

        vocab.append(token)

        probs.append(

            p[idx].item()

        )

        ids.append(idx)

    return {

        "token":vocab,

        "probability":probs,

        "id":ids

    }


def layer_embedding_matrix(outputs):

    mat=[]

    for h in outputs.hidden_states:

        mat.append(

            h[0,-1].cpu().numpy()

        )

    return np.stack(mat)



def extract_from_outputs(inputs, outputs):

    signals = {}

    signals["ids"] = extract_ids(inputs)

    signals["tokens"] = extract_tokens(inputs)

    signals["last_hidden"] = np.stack(

        extract_last_hidden(outputs),

        axis=0

    )

    signals["layer_matrix"] = signals["last_hidden"]

    signals["hidden"] = np.stack(

        extract_hidden(outputs),

        axis=0

    )

    signals["token_hidden"] = signals["hidden"]

    signals["probability"] = extract_probability(outputs)

    signals["entropy"] = extract_entropy(outputs)

    signals["attention"] = extract_attention(outputs)

    signals["rollout"] = extract_rollout(outputs)

    signals["logits"] = extract_logits(outputs)

    signals["logit_lens"] = extract_logit_lens(outputs)

    return signals


def cache_signal(

    signal,

    cache_path

):

    with open(

        cache_path,

        "wb"

    ) as f:

        pickle.dump(

            signal,

            f,

            protocol=pickle.HIGHEST_PROTOCOL

        )

def extract_everything(

        full_text,

        cache_path=None,

        overwrite=False

):

    if (

        cache_path is not None

        and

        os.path.exists(cache_path)

        and

        not overwrite

    ):

        return load_signal(cache_path)

    inputs, outputs = forward(full_text)

    signals = extract_from_outputs(

        inputs,

        outputs

    )

    if cache_path is not None:

        cache_signal(

            signals,

            cache_path

        )

    return signals
