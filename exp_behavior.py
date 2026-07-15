"""
Experiment 1

Behavior Analysis

Fig1

Accuracy

Fig2

Answer Consistency

Fig3

Behavior Divergence
"""

import os
import re
import numpy as np
import pandas as pd

from tqdm import tqdm

from metrics import *

from extraction import *

from config import *

#############################################################
# Utils
#############################################################

def extract_final_number(text):

    nums=re.findall(

        r"[-+]?\d*\.?\d+",

        text.replace(",","")

    )

    if len(nums)==0:

        return None

    return float(nums[-1])

#############################################################
# Accuracy
#############################################################

def evaluate_behavior():

    index=pd.read_csv(

        os.path.join(

            CACHE_DIR,

            "cache_index.csv"

        )

    )

    results=[]

    #######################################################

    for _,row in tqdm(

            index.iterrows(),

            total=len(index)

    ):

        gt=extract_final_number(

            str(

                row["ground_truth"]

            )

        )

        prompts=[]

        answers=[]

        acc=[]

        ###################################################

        for prompt in PROMPT_FAMILIES.keys():

            cot=row[prompt+"_answer"]

            ans=extract_final_number(

                str(cot)

            )

            prompts.append(prompt)

            answers.append(ans)

            if gt is None:

                acc.append(0)

            else:

                acc.append(

                    int(ans==gt)

                )

        ###################################################

        consistency=len(

            set(answers)

        )==1

        ###################################################

        results.append(

            {

                "id":row["id"],

                "accuracy_mean":np.mean(acc),

                "answer_consistency":consistency,

                "accuracy_std":np.std(acc)

            }

        )

    df=pd.DataFrame(results)

    df.to_csv(

        os.path.join(

            RESULT_DIR,

            "behavior.csv"

        ),

        index=False

    )

    print(df.describe())

    return df
