"""
Cache Builder

Generate all reasoning traces and save every signal.

Only run ONCE.
"""

import os
import pickle
import pandas as pd

from tqdm import tqdm

from config import *

from prompts import *

from generation import *

from extraction import *

#############################################################
# Cache Folder
#############################################################

SIGNAL_CACHE = os.path.join(

    CACHE_DIR,

    "signals"

)

os.makedirs(

    SIGNAL_CACHE,

    exist_ok=True

)

#############################################################
# Build Cache
#############################################################

def build_signal_cache(

        dataset,

        prompt_family=PROMPT_FAMILIES,

        overwrite=False

):

    index=[]

    for idx,sample in enumerate(

            tqdm(dataset)

    ):

        question=sample["question"]

        answer=sample["answer"]

        base_prompt=tokenizer.apply_chat_template(

            [

                {

                    "role":"user",

                    "content":question

                }

            ],

            tokenize=False,

            add_generation_prompt=True

        )

        sample_record={

            "id":idx,

            "question":question,

            "ground_truth":answer

        }

        ###################################################
        # Every Prompt Family
        ###################################################

        for prompt_name,prompt in prompt_family.items():

            cot=generate_reasoning(

                base_prompt,

                prompt

            )

            full_text=(

                base_prompt

                +prompt

                +cot

                +FINAL_PHRASE

            )

            cache_file=os.path.join(

                SIGNAL_CACHE,

                f"{idx:05d}_{prompt_name}.pkl"

            )

            signal=extract_everything(

                full_text,

                cache_path=cache_file,

                overwrite=overwrite

            )

            sample_record[prompt_name]=cache_file

            sample_record[prompt_name+"_answer"]=cot

        index.append(

            sample_record

        )

    df=pd.DataFrame(index)

    df.to_csv(

        os.path.join(

            CACHE_DIR,

            "cache_index.csv"

        ),

        index=False

    )

    print()

    print("Cache Finished.")

    print(df.head())

    return df
