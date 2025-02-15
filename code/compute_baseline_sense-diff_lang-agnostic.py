import os
os.environ['TRANSFORMERS_CACHE'] = dir_cache
os.environ['HF_HOME'] = dir_cache
os.environ['HF_DATASETS_CACHE'] = dir_cache
os.environ['TORCH_HOME'] = dir_cache

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm
from itertools import product
from huggingface_hub import login
huggingface_token = ""
login(token=huggingface_token)
import pandas as pd
import sys
dir_wd = ""
sys.path.append(dir_wd)
from utils import compute_sent_hidden_states, compute_baseline_en, compute_baseline_zh
import click

dir_results = os.path.join(dir_wd, "results", "baseline_v20250118")
os.makedirs(dir_results, exist_ok=True)

@click.command()
@click.option('-m', '--model-name', required=True, help="The model name on hf")
@click.option('-l', '--lang', required=True, help="zh/en")
def main(model_name, lang):
    
    fname_output = f"{dir_results}{os.sep}df_baseline_ambiguity-lexical_lang-{lang}_sense-diff_model-{model_name.split('/')[-1]}"
    if os.path.exists(fname_output):
        click.echo("Results already exist")
        return

    df_homonymy = pd.read_excel(f"{dir_wd}{os.sep}lexical_ambiguity{os.sep}df_homonymy_v20251117.xlsx", sheet_name=lang)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    this_model = AutoModel.from_pretrained(model_name).to(device)
    this_tokenizer = AutoTokenizer.from_pretrained(model_name)

    list_df = []
    list_target_words = df_homonymy.Word.unique()
    for target_word in list_target_words:
        list_sent_1 = df_homonymy.loc[df_homonymy.Word==target_word]['Sentence1'].values
        list_sent_2 = df_homonymy.loc[df_homonymy.Word==target_word]['Sentence2'].values
        n_sent = len(list_sent_1)

        # Compute the hidden states of all sentences belonging to this target word
        list_hidden_states_1 = compute_sent_hidden_states(this_tokenizer, this_model, list_sent_1)
        list_hidden_states_2 = compute_sent_hidden_states(this_tokenizer, this_model, list_sent_2)
        
        comb_sent_idx = list(product(range(n_sent), range(n_sent)))
        for this_comb in comb_sent_idx:
            input_sent_1 = list_sent_1[this_comb[0]]
            input_sent_2 = list_sent_2[this_comb[1]]
            this_hidden_states_1 = list_hidden_states_1[this_comb[0]] # hidden_states of the this_comb[0] sentence in list 1
            this_hidden_states_2 = list_hidden_states_2[this_comb[1]] # hidden_states of the this_comb[1] sentence in list 2

            match lang:
                case "en":
                    this_df = compute_baseline_en(this_tokenizer, input_sent_1, input_sent_2, this_hidden_states_1, this_hidden_states_2, target_word=target_word, n_iter=30)
                case "zh":
                    this_df = compute_baseline_zh(this_tokenizer, input_sent_1, input_sent_2, this_hidden_states_1, this_hidden_states_2, target_word=target_word, n_iter=30)
            this_df["idx_sent_1"] = this_comb[0]
            this_df["idx_sent_2"] = this_comb[1]
            list_df.append(this_df)
        print(f"{model_name} | {target_word}")

    df = pd.concat(list_df)
    df["angle"] = np.arccos(df["cos_sim"]) * 180 / np.pi
    df_mean = df.groupby(by=[df.target_word, df.idx_sent_1, df.idx_sent_2, df.layer]).agg({"cos_sim": "mean", "angle": "mean", "layer_rel": "first"}).assign(model=model_name.split('/')[-1]).reset_index()
    
    df.to_feather(f"{fname_output}_raw")
    df_mean.to_feather(fname_output)

if __name__ == "__main__":
    main()