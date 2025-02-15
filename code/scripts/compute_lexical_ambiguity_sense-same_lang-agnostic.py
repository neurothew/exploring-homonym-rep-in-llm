import os
dir_cache = ""
# os.environ['TRANSFORMERS_CACHE'] = dir_cache
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

from utils import compute_word_cossim
from huggingface_hub import login
huggingface_token = ""
login(token=huggingface_token)

@click.command()
@click.option('-m', '--model-name', required=True, help="The model name on hf")
@click.option('-l', '--lang', required=True, help="en/zh")
def main(model_name, lang):
    fname_output = f"{dir_results}{os.sep}df_ambiguity-lexical_lang-{lang}_sense-same_model-{model_name.split('/')[-1]}"
    if os.path.exists(fname_output):
        print("Results already exist.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    this_model = AutoModel.from_pretrained(model_name).to(device) # This will load the raw pre-trained model without adding any additional layers
    click.echo(f"Finish loading model {model_name}")
    this_tokenizer = AutoTokenizer.from_pretrained(model_name)
    import pandas as pd
    # install openpyxl to the environment if you cannot read the excel file
    df_homonymy = pd.read_excel(f"{dir_wd}{os.sep}lexical_ambiguity{os.sep}df_homonymy_v20251117.xlsx", sheet_name=lang)
    list_word_of_interest = df_homonymy.Word.unique()

    list_df_cossim_word_all = []
    pbar = tqdm(list_word_of_interest, desc="Processing words")
    for target_word in pbar:
        pbar.set_postfix({'word': target_word, "model": model_name})
        list_df_cossim_word = []
        list_sent_1 = df_homonymy.loc[df_homonymy.Word==target_word]["Sentence1"].values
        list_sent_2 = df_homonymy.loc[df_homonymy.Word==target_word]["Sentence2"].values
        lists_sent = [list_sent_1]
        df_cos_sim = compute_word_cossim(this_model, this_tokenizer, lists_sent, target_word, lang=lang, is_custom_model=False, model_name=model_name)
        df_cos_sim["sense"] = 1
        list_df_cossim_word.append(df_cos_sim.reset_index(drop=True))

        lists_sent = [list_sent_2]
        df_cos_sim = compute_word_cossim(this_model, this_tokenizer, lists_sent, target_word, lang=lang, is_custom_model=False, model_name=model_name)
        df_cos_sim["sense"] = 2
        list_df_cossim_word.append(df_cos_sim.reset_index(drop=True))

        df_cossim_word = pd.concat(list_df_cossim_word).reset_index(drop=True)

        df_this_property = df_homonymy.loc[df_homonymy.Word==target_word][["Word", "POS1", "POS2", "SamePOS"]].head(1)
        df_this_property = df_this_property.rename(columns={"Word": "word"})
        df_cossim_word = pd.merge(df_cossim_word, df_this_property, on="word")

        list_df_cossim_word_all.append(df_cossim_word)

    df_cossim_word_all = pd.concat(list_df_cossim_word_all).reset_index(drop=True)
    df_cossim_word_all_mean = df_cossim_word_all.groupby(["word", "sense", "layer", "POS1", "POS2", "SamePOS"]).agg({"cos_sim": "mean", "angle": "mean", "layer_rel": "first"}).assign(model=model_name.split('/')[-1]).reset_index()

    df_cossim_word_all.to_feather(f"{fname_output}_raw")
    df_cossim_word_all_mean.to_feather(fname_output)


if __name__ == "__main__":
    main()