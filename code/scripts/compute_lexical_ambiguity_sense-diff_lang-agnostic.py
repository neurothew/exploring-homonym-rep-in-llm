import os
dir_cache = ""
os.environ['TRANSFORMERS_CACHE'] = dir_cache
os.environ['HF_HOME'] = dir_cache
os.environ['HF_DATASETS_CACHE'] = dir_cache
os.environ['TORCH_HOME'] = dir_cache

import torch
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm
import pandas as pd
import sys
import click
dir_wd = ""
sys.path.append(dir_wd)
dir_results = os.path.join(dir_wd, "results", "lexical_v20250118")

from code.scripts.utils import compute_word_cossim
from huggingface_hub import login
huggingface_token = ""
login(token=huggingface_token)

@click.command()
@click.option('-m', '--model-name', required=True, help="The model name on hf")
@click.option('-l', '--lang', required=True, help="en/zh")
def main(model_name, lang):
    fname_output = f"{dir_results}{os.sep}df_ambiguity-lexical_lang-{lang}_sense-diff_model-{model_name.split('/')[-1]}"
    if os.path.exists(fname_output):
        print("Results already exist.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    this_model = AutoModel.from_pretrained(model_name).to(device) # This will load the raw pre-trained model without adding any additional layers
    click.echo(f"Finish loading model {model_name}")
    this_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # install openpyxl to the environment if you cannot read the excel file
    df_homonymy = pd.read_excel(f"{dir_wd}{os.sep}lexical_ambiguity{os.sep}df_homonymy_v20251117.xlsx", sheet_name=lang)
    list_word_of_interest = df_homonymy.Word.unique()

    list_all_df_cos_sim = []
    pbar = tqdm(list_word_of_interest, desc="Processing words")

    for target_word in pbar:
        pbar.set_postfix({'word': target_word, "model": model_name})
        list_sent_1 = df_homonymy.loc[df_homonymy.Word==target_word]["Sentence1"].values
        list_sent_2 = df_homonymy.loc[df_homonymy.Word==target_word]["Sentence2"].values
        
        lists_sent = [list_sent_1, list_sent_2]    
        
        df_cos_sim = compute_word_cossim(this_model, this_tokenizer, lists_sent, target_word, lang=lang, is_custom_model=False, model_name=model_name)
        
        if type(df_cos_sim) != pd.DataFrame: # Something goes wrong during get_word_representations, could be caused by UNK token in BERT-related architecture
            df_cos_sim = pd.DataFrame([{"word": target_word}])
            # continue

        df_this_property = df_homonymy.loc[df_homonymy.Word==target_word][["Word", "POS1", "POS2", "SamePOS"]].head(1)
        df_this_property = df_this_property.rename(columns={"Word": "word"})

        df_cos_sim = pd.merge(df_cos_sim, df_this_property, on="word")

        list_all_df_cos_sim.append(df_cos_sim)

    df_cos_sim_all = pd.concat(list_all_df_cos_sim)
    df_cos_sim_all_mean = df_cos_sim_all.groupby(["word", "layer", "POS1", "POS2", "SamePOS"]).agg({"cos_sim": "mean", "angle": "mean", "layer_rel": "first"}).assign(model=model_name.split('/')[-1]).reset_index()

    df_cos_sim_all.to_feather(f"{fname_output}_raw")
    df_cos_sim_all_mean.to_feather(fname_output)

if __name__ == "__main__":
    main()