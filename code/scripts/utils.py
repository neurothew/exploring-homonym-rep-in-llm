import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import re
from torch.nn import CosineSimilarity
import os
from tqdm import tqdm
from itertools import product
import pandas as pd
import random
import itertools
import pdb
from functools import wraps
import jieba

def list_files(datadir, pattern):
    """ List files with desired pattern using regular expression
    Parameters
    --------
    datadir: str
        Data directory
    pattern: str 
        Pattern in regular expression
    
    Returns
    --------
    filelist: list
        list of files with desired pattern.
    """
    filelist = [f for f in os.listdir(datadir) if re.match(pattern, f)]
    filelist = [os.path.join(datadir, filename) for filename in filelist]
    return (filelist)

def get_bids_fname_tag(fname, tag):
    """
    Return the value of a particular BIDS tag.

    Return
    ------
    >>> fname = "df_task-rest"
    >>> get_bids_fname_tag(fname, "task")
    >>> "rest"
    """
    re_pattern = f"{tag}-[a-zA-Z0-9\-]*"
    thisre = re.search(re_pattern, fname).group()
    if len(re.findall('-', thisre)) == 1:
        thisattr = thisre.split('-')[-1]
    elif len(re.findall('-', thisre)) > 1: #preserve all later parts
        thisattr = "-".join(thisre.split('-')[1:len(thisre)])
    return thisattr


def debugging(debug=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def print_debug(*msg):
                if debug:
                    print(*msg)
                # When debug is False, this function does nothing
            
            # Add print_debug to the function's local namespace
            func.__globals__['print_debug'] = print_debug
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Remove print_debug from the namespace after function execution
                func.__globals__.pop('print_debug', None)
            
            return result
        return wrapper
    return decorator

def compute_manual_hidden_states(custom_model, this_tokenizer, this_input_text):
    """
    Compute the hidden representations/hidden states/embeddings step by step with a custom model.
    Currently only BERT is accepted.

    Parameters
    ----------
    custom_model: transformer model
        Transformer model from AutoModel/BertModel
    this_tokenizer: transformer tokenizer
        Tokenizer from AutoTokenizer/BertTokenizer
    this_input_text: str
        The input text

    Example
    -------
    >>> model_name = 'bert-base-uncased'
    >>> bert_model = BertModel.from_pretrained(model_name)
    >>> tokenizer = BertTokenizer.from_pretrained(model_name)
    >>> text = "Hello, I am using this function."
    >>> manual_hidden_states = compute_manual_hidden_states(bert_model, tokenizer, text)
    """
    inputs = this_tokenizer(this_input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    with torch.no_grad():
        hidden_states = custom_model.embeddings(input_ids)
        manual_hidden_states = [hidden_states]
        # print(hidden_states.size())
        for layer in custom_model.encoder.layer:
            attention_output = layer.attention.self(hidden_states)
            attention_output = layer.attention.output(attention_output[0], hidden_states) # The first hidden states is the word embedding
            intermediate_output = layer.intermediate(attention_output)
            layer_output = layer.output(intermediate_output, attention_output)
            hidden_states = layer_output
            # print(attention_output.size(), intermediate_output.size(), layer_output.size())
            manual_hidden_states = manual_hidden_states + [hidden_states]
        return(manual_hidden_states)

def rand_weight_cols(weight, rand_per):
    """
    Given a weight matrix, randomize a particular percentage of columns.

    The randomized weights are scaled to have similar mean and sd as the original weights

    Parameters
    ----------
    weight: torch.tensor
        The weight matrix from any layer components
    rand_per: float
        Percentage of columns to be randomized

    Example
    ------- 
    >>> ori_weight = test_model.encoder.layer[target_layer_idx].attention.output.dense.weight
    >>> updated_weights = rand_weight_cols(ori_weight, rand_per)
    >>> ori_weight.data.copy_(updated_weights)
    """
    updated_weights = weight.clone()
    
    n_dim = updated_weights.shape[1]
    n_dim_to_rand = int(n_dim * rand_per)
    
    # Randomly select columns to randomize
    cols_to_rand = random.sample(range(n_dim), n_dim_to_rand)
    
    # Create a mask for the selected columns
    mask = torch.zeros(n_dim, dtype=torch.bool)
    mask[cols_to_rand] = True
    

    # Calculate mean and std of the original tensor
    ori_mean = weight.mean().item()
    ori_std = weight.std().item()
    
    # Generate random values for the selected columns
    rand_cols = torch.randn_like(updated_weights[:, mask])
    
    # Scale the random values to match the original distribution
    min_val, max_val = weight.min().item(), weight.max().item()
    ori_range = max_val - min_val
    rand_range = rand_cols.max().item() - rand_cols.min().item()
    
    scaled_rand_cols = (rand_cols - rand_cols.min().item()) / rand_range * ori_range + min_val
    
    # Adjust mean and std to match original
    scaled_rand_cols = (scaled_rand_cols - scaled_rand_cols.mean()) / scaled_rand_cols.std() * ori_std + ori_mean

    # # Calculate mean and std of the original tensor
    # ori_mean = weight.mean().item()
    # ori_std = weight.std().item()
    
    # # Generate random values for the selected columns with matching statistics
    # rand_cols = torch.randn_like(updated_weights[:, mask]) * ori_std + ori_mean
    
    # # Clip the values to match the range of the original tensor
    # min_val, max_val = weight.min().item(), weight.max().item()
    # rand_cols.clamp_(min_val, max_val)
    
    # # Update the selected columns with random values
    updated_weights[:, mask] = scaled_rand_cols
    return updated_weights

def get_word_embeddings_preload_arr(this_model, this_tokenizer, input_sent, target_word, is_custom_model=False):
    """
    Extract the hidden representations of the target word at each layer.
    
    Parameter
    ---------
    this_model: Transformer model
        The PLM/LLM
    this_tokenizer: Transformer tokenizer
        Corresponding tokenizer of the PLM/LLM
    input_sent: str
        The input sentence
    target_word: str
        The target word

    Return
    ------
    this_embedding_arr: np.ndarray (n_layer, n_dim)
        The hidden representation of the target word across all layers
    """
    # After loading the model and tokenizer, we want to encode our input sentence.
    # By encoding, that means to tokenize the input sentence and convert each token to the corresponding ids in the vocabulary of our model of interest.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tokens = this_tokenizer.encode_plus(input_sent, return_tensors="pt")
    input_tokens = input_tokens.to(device)
    # Deprecated old codes: Find all words and punctuation
    # pattern_words = r'\w+|[^\w\s]'
    # list_words = re.findall(pattern_words, input_sent)
    # print(list_words)
    # idx_word = list_words.index(target_word)

    list_words = get_list_words(this_tokenizer, input_sent)
    idx_word = get_idx_word_id(list_words, target_word)

    # Since idx_word is identified by assuming that list_words is exactly how the tokenizer determine the list of words
    # Cannot use regex to obtain the list of words.
    idx_word_tokens = np.where(np.array(input_tokens.word_ids()) == idx_word)[0]

    if is_custom_model:
        with torch.no_grad():
            manual_hidden_states = compute_manual_hidden_states(this_model, this_tokenizer, input_sent)
            # TODO: to map manual_hidden_states back to cpu
            this_embedding_arr = np.array(manual_hidden_states)[:, 0, :, :][:, idx_word_tokens, :].mean(axis=1)
    else:
        with torch.no_grad():
            outputs = this_model(**input_tokens, output_hidden_states=True) 
        hidden_states_cpu = tuple(map(lambda x: x.to('cpu'), outputs.hidden_states))
        this_embedding_arr = np.array(hidden_states_cpu)[:, 0, :, :][:, idx_word_tokens, :].mean(axis=1)

    return(this_embedding_arr)

def get_word_representations(this_model, this_tokenizer, input_sent, target_word, lang='en', is_custom_model=False, model_name=None):
    """
    Extract the hidden representations of the target word at each layer.
    
    Language-agnostic

    Parameter
    ---------
    this_model: Transformer model
        The PLM/LLM
    this_tokenizer: Transformer tokenizer
        Corresponding tokenizer of the PLM/LLM
    input_sent: str
        The input sentence
    target_word: str
        The target word
    is_zh: bool
        If true, use `get_idx_chinese_word` to retrieve idx_word_tokens
    Return
    ------
    this_embedding_arr: np.ndarray (n_layer, n_dim)
        The hidden representation of the target word across all layers
    """
    # After loading the model and tokenizer, we want to encode our input sentence.
    # By encoding, that means to tokenize the input sentence and convert each token to the corresponding ids in the vocabulary of our model of interest.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Deprecated old codes: Find all words and punctuation
    # pattern_words = r'\w+|[^\w\s]'
    # list_words = re.findall(pattern_words, input_sent)
    # print(list_words)
    # idx_word = list_words.index(target_word)
    input_tokens = this_tokenizer.encode_plus(input_sent, return_tensors="pt")
    input_tokens = input_tokens.to(device)

    match lang:
        case "en":
            list_words = get_list_words(this_tokenizer, input_sent)
            idx_word = get_idx_word_id(list_words, target_word)
            # Since idx_word is identified by assuming that list_words is exactly how the tokenizer determine the list of words
            # Cannot use regex to obtain the list of words.
            idx_word_tokens = np.where(np.array(input_tokens.word_ids()) == idx_word)[0]
        case "zh":
            this_word_dict = get_idx_zh_word(this_tokenizer, input_sent, target_word, is_debug=False, model_name=model_name)
            idx_word_tokens = np.arange(this_word_dict['idx_start_char1'], this_word_dict['idx_end_char2']+1)
            # if this_word_dict["error"] == 0:
            #     idx_word_tokens = np.arange(this_word_dict['idx_start_char1'], this_word_dict['idx_end_char2']+1)
            # else:
            #     return 0
        case _:
            print("The input language is not supported yet. Please use 'en' or 'zh'")
            return 0

    if is_custom_model:
        pass
        # with torch.no_grad():
        #     manual_hidden_states = compute_manual_hidden_states(this_model, this_tokenizer, input_sent)
        #     # TODO: to map manual_hidden_states back to cpu
        #     this_embedding_arr = np.array(manual_hidden_states)[:, 0, :, :][:, idx_word_tokens, :].mean(axis=1)
    else:
        with torch.no_grad():
            outputs = this_model(**input_tokens, output_hidden_states=True) 
        hidden_states_cpu = tuple(map(lambda x: x.to('cpu'), outputs.hidden_states))
        this_embedding_arr = np.array(hidden_states_cpu)[:, 0, :, :][:, idx_word_tokens, :].mean(axis=1)

    return(this_embedding_arr)

def compute_cos_sim_df_word(this_model, this_tokenizer, list_sent_1, list_sent_2, target_word, lang='en', is_custom_model=False, model_name=None):
    n_sent = len(list_sent_1)
    list_embedding_arr_1 = []
    list_embedding_arr_2 = []
    for i in range(n_sent):
        try:
            this_embedding_arr_1 = get_word_representations(this_model, this_tokenizer, list_sent_1[i], target_word, lang=lang, is_custom_model=is_custom_model, model_name=model_name)
            this_embedding_arr_2 = get_word_representations(this_model, this_tokenizer, list_sent_2[i], target_word, lang=lang, is_custom_model=is_custom_model, model_name=model_name)
        except:
            # pdb.set_trace()
            print(f"Errors while computing embedding for word: {target_word}")
            return 0
        list_embedding_arr_1.append(this_embedding_arr_1)
        list_embedding_arr_2.append(this_embedding_arr_2)

    list_dict = []
    for this_comb in list(product(range(n_sent), range(n_sent))):
        try:
            torch_arr_1 = torch.from_numpy(list_embedding_arr_1[this_comb[0]])
            torch_arr_2 = torch.from_numpy(list_embedding_arr_2[this_comb[1]])
            cos_sim = CosineSimilarity(dim=1)
            this_cos_sim = cos_sim(torch_arr_1, torch_arr_2)
            for i in range(this_cos_sim.shape[0]):
                this_dict = {
                    "idx_sent_1": this_comb[0], # note that I intentionally make it 1-indexed for better readability
                    "idx_sent_2": this_comb[1], # same as above
                    "sent1": list_sent_1[this_comb[0]],
                    "sent2": list_sent_2[this_comb[1]],
                    "layer": i,
                    "layer_rel": i/(this_cos_sim.shape[0]-1),
                    "cos_sim": this_cos_sim[i].item()
                }
                list_dict.append(this_dict)
        except:
            print(f"Some errors in computing cosine similarity for {target_word} | {list_sent_1[this_comb[0]]} | {list_sent_1[this_comb[1]]}")
            continue
    df_cos_sim = pd.DataFrame(list_dict).reset_index()
    df_cos_sim['word'] = target_word
    return(df_cos_sim)


def gen_all_pair_list(list_words_1: list, list_words_2: list, target_word: str) -> list:
    """
    Given two lists of words, generate a list of pairs of words, except the pair [target_word, target_word].

    Parameters
    ----------
    list_words_1: list
        list of words for sentence 1
    list_words_2: list
        list of words for sentence 2
    target_word: str
        the target word
    
    Example
    -------
    >>> list_all_pairs = gen_all_pair_list(list_words_1, list_words_2, "fan")
    """
    exclude_pair = [target_word, target_word]
    return [pair for pair in itertools.product(list_words_1, list_words_2) if pair != exclude_pair]

def gen_random_pair_list(list1, list2):
    num_pairs = min(len(list1), len(list2))
    temp_list1 = list1.copy()
    temp_list2 = list2.copy()
    
    random.shuffle(temp_list1)
    random.shuffle(temp_list2)
    return list(zip(temp_list1[:num_pairs], temp_list2[:num_pairs]))

def draw_random_word_pair(list_words_1: list, list_words_2: list, target_word: str) -> tuple:
    exclude_pair = [target_word, target_word]
    is_not_exclude_pair = False
    if not is_not_exclude_pair:
        word1 = random.choice(list_words_1)
        word2 = random.choice(list_words_2)
        if (word1 != target_word) and (word2 != target_word):
            is_not_exclude_pair = True
    return (word1, word2)

def compute_cross_sent_word_similarity(this_tokenizer, input_sent_1, input_sent_2, hidden_states_1, hidden_states_2, target_word, n_iter=50) -> pd.DataFrame:
    """
    Generate dataframe storing the cosine similarity between random words (including punctuations) given two input sentences.

    Parameters
    ----------
    this_tokenizer: Tokenizer
        tokenizer from the transformers library
    input_sent_1: str
        Sentence 1
    input_sent_2: str
        Sentence 2
    hidden_states_1: numpy.ndarray (n_layer, n_sent, n_words, n_dim)
        The hidden_states returned from encoding sentence 1
    hidden_states_2: numpy.ndarray (n_layer, n_sent, n_words, n_dim)
        The hidden_states returned from encoding sentence 2
    target_word: str
        The target word
    n_iter:
        How many pairs? or how many iterations?

    Example
    -------
    >>> this_df = gen_df_random_cos_sim(input_sent_1, input_sent_2, this_hidden_states_1, this_hidden_states_2, target_word=this_target_word, n_iter=50)

    """
    # Find all words and punctuation
    regex_words = r'\w+|[^\w\s]'
    # regex_words = r'\w+'

    list_words_1 = re.findall(regex_words, input_sent_1)
    list_words_2 = re.findall(regex_words, input_sent_2)
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tokens_1 = this_tokenizer.encode_plus(input_sent_1, return_tensors="pt")
        input_tokens_2 = this_tokenizer.encode_plus(input_sent_2, return_tensors="pt")
        input_tokens_1 = input_tokens_1.to(device)
        input_tokens_2 = input_tokens_2.to(device)
        
        list_dict = []

        # This results in the list of all possible pairs
        list_all_pairs = gen_all_pair_list(list_words_1, list_words_2, "fan")
        random.shuffle(list_all_pairs)
        for this_pair in list_all_pairs[0:n_iter]:
        # for idx_iter in range(0, n_iter):
            # this_pair = draw_random_word_pair(list_words_1, list_words_2, target_word)
            idx_word_1 = list_words_1.index(this_pair[0])
            idx_word_2 = list_words_2.index(this_pair[1])
            # print(idx_word_1, idx_word_2)
            # print(list_words_1[idx_word_1], list_words_2[idx_word_2])
            idx_word_tokens_1 = np.where(np.array(input_tokens_1.word_ids()) == idx_word_1)[0]
            idx_word_tokens_2 = np.where(np.array(input_tokens_2.word_ids()) == idx_word_2)[0]
            try:
                hidden_states_word_1 = np.array(hidden_states_1)[:, 0, :, :][:, idx_word_tokens_1, :].mean(axis=1)
                hidden_states_word_2 = np.array(hidden_states_2)[:, 0, :, :][:, idx_word_tokens_2, :].mean(axis=1)
            except:
                pdb.set_trace()

            torch_arr_1 = torch.from_numpy(hidden_states_word_1)
            torch_arr_2 = torch.from_numpy(hidden_states_word_2)
            cos_sim = CosineSimilarity(dim=1)
            this_cos_sim = cos_sim(torch_arr_1, torch_arr_2)
            for i in range(this_cos_sim.shape[0]):
                this_dict = {
                    "sent1": input_sent_1,
                    "sent2": input_sent_2,
                    "target_word": target_word,
                    "word1": this_pair[0],
                    "word2": this_pair[1],      
                    "layer": i,
                    "layer_rel": i/(this_cos_sim.shape[0]-1),
                    "cos_sim": this_cos_sim[i].item()
                }
                list_dict.append(this_dict)
        return(pd.DataFrame(list_dict)) # this dataframe carries no information on the index in the combination

def compute_cross_sent_word_similarity2(this_tokenizer, input_sent_1, input_sent_2, hidden_states_1, hidden_states_2, target_word, n_iter=50) -> pd.DataFrame:
    """
    Generate dataframe storing the cosine similarity between random words (including punctuations) given two input sentences.

    Changelog
    ---------
    - Change gen_all_pair_list to gen_all_pair_idx_list

    Parameters
    ----------
    this_tokenizer: Tokenizer
        tokenizer from the transformers library
    input_sent_1: str
        Sentence 1
    input_sent_2: str
        Sentence 2
    hidden_states_1: numpy.ndarray (n_layer, n_sent, n_words, n_dim)
        The hidden_states returned from encoding sentence 1
    hidden_states_2: numpy.ndarray (n_layer, n_sent, n_words, n_dim)
        The hidden_states returned from encoding sentence 2
    target_word: str
        The target word
    n_iter:
        How many pairs? or how many iterations?

    Example
    -------
    >>> this_df = gen_df_random_cos_sim(input_sent_1, input_sent_2, this_hidden_states_1, this_hidden_states_2, target_word=this_target_word, n_iter=50)

    """
    # Find all words and punctuation
    # regex_words = r'\w+|[^\w\s]'

    # list_words_1 = re.findall(regex_words, input_sent_1)
    # list_words_2 = re.findall(regex_words, input_sent_2)

    list_words_1 = get_list_words(this_tokenizer, input_sent_1)
    list_words_2 = get_list_words(this_tokenizer, input_sent_2)
    
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tokens_1 = this_tokenizer.encode_plus(input_sent_1, return_tensors="pt")
        input_tokens_2 = this_tokenizer.encode_plus(input_sent_2, return_tensors="pt")
        input_tokens_1 = input_tokens_1.to(device)
        input_tokens_2 = input_tokens_2.to(device)
        
        list_dict = []

        # This results in the list of all possible pairs
        # list_all_pairs = gen_all_pair_list(list_words_1, list_words_2, target_word)
        list_all_pairs = gen_all_pair_idx_list(list_words_1, list_words_2, target_word)
        random.shuffle(list_all_pairs)
        for this_pair in list_all_pairs[0:n_iter]:
            try:
                idx_word_tokens_1 = np.where(np.array(input_tokens_1.word_ids()) == this_pair[0])[0]
                idx_word_tokens_2 = np.where(np.array(input_tokens_2.word_ids()) == this_pair[1])[0]
            except:
                print("Something happening...")
            if (len(idx_word_tokens_2)==0) or (len(idx_word_tokens_1)==0):
                # There are exceptional cases in which the sentence may contain something like "the teacher's", 
                # then the word ids will be incorrectly recognized by the regex pattern... 
                print("Hello World")

            # hidden_states_1_cpu = tuple(map(lambda x: x.to('cpu'), hidden_states_1))
            # hidden_states_2_cpu = tuple(map(lambda x: x.to('cpu'), hidden_states_2))
            # pdb.set_trace()
            hidden_states_word_1 = np.array(hidden_states_1)[:, 0, :, :][:, idx_word_tokens_1, :].mean(axis=1)
            hidden_states_word_2 = np.array(hidden_states_2)[:, 0, :, :][:, idx_word_tokens_2, :].mean(axis=1)
            
            torch_arr_1 = torch.from_numpy(hidden_states_word_1)
            torch_arr_2 = torch.from_numpy(hidden_states_word_2)
            cos_sim = CosineSimilarity(dim=1)
            this_cos_sim = cos_sim(torch_arr_1, torch_arr_2)
            for i in range(this_cos_sim.shape[0]):
                this_dict = {
                    "sent1": input_sent_1,
                    "sent2": input_sent_2,
                    "target_word": target_word,
                    "word1_idx_in_list_words_1": this_pair[0],
                    "word2_idx_in_list_words_2": this_pair[1],
                    "word1": list_words_1[this_pair[0]],
                    "word2": list_words_2[this_pair[1]],      
                    "layer": i,
                    "layer_rel": i/(this_cos_sim.shape[0]-1),
                    "cos_sim": this_cos_sim[i].item()
                }
                list_dict.append(this_dict)
        return(pd.DataFrame(list_dict)) # this dataframe carries no information on the index in the combination


def compute_sent_hidden_states(this_tokenizer, this_model, list_sent: list) -> list:
    """
    Compute the hidden representation of each sentence in the list, and return a list of hidden representations.

    Parameters
    ----------
    this_tokenizer: Tokenizer
        Toeknizer from transformer library
    this_model: Model
        Language model from transformer library
    list_sent: list
        list of sentences

    Return
    ------
    list_hidden_states: list
        A list of np.ndarray of size (n_layer, n_sent, n_words, n_dim)

    Example
    -------
    >>> compute_sent_hidden_states(this_tokenizer, this_model, list_sent)
    """
    list_hidden_states = []
    for i in range(len(list_sent)):
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_tokens = this_tokenizer.encode_plus(list_sent[i], return_tensors="pt")
            input_tokens = input_tokens.to(device)
            outputs = this_model(**input_tokens, output_hidden_states=True)
            hidden_states_cpu = tuple(map(lambda x: x.to('cpu'), outputs.hidden_states))
            hidden_states = np.array(hidden_states_cpu)

        list_hidden_states.append(hidden_states)
    return list_hidden_states



def get_list_words(this_tokenizer, this_sent, is_debug=False):
    """
    Ver: 20241003
    Given a tokenizer and a sentence, return a list of words.

    Note that the "word" here refers to what's being recognized as the same word_id, it doesn't necessarily mean the intuitive sense of word.

    Parameter
    ---------
    this_tokenizer:
        Transformer tokenizer
    this_sent: str
        Target sentence

    Return
    ------
    list_words: list
        A list of words 
    
    """
    this_encode = this_tokenizer.encode_plus(this_sent)
    this_word_ids_without_none = [this_word_id for this_word_id in this_encode.word_ids() if this_word_id != None]
    list_token_ids = this_encode['input_ids']

    if is_debug:
        # This is the decoded token
        list_decode_tokens = []
        for this_input_id in list_token_ids:
            list_decode_tokens.append(this_tokenizer.decode(this_input_id))
        print(list_decode_tokens)

    # This is the decoded word, with the definition of word defined by word_ids()
    list_words = []
    for i, this_word_id in enumerate(this_word_ids_without_none):
        this_tokenspan = this_encode.word_to_tokens(this_word_id)
        this_word = this_tokenizer.decode(list_token_ids[this_tokenspan.start:this_tokenspan.end])
        if (len(list_words) >= 1) and (this_word == list_words[-1]):
            # print(f"{this_word} appears more than twice in a row")
            pass
        else:
            list_words.append(this_word)
    return(list_words)

def get_idx_word_id(list_words, target_word):
    """
    Ver: 20241003
    """
    list_find_idx = []
    for this_word in list_words:
        list_matching = re.findall(f"{re.escape(target_word)}", this_word.strip(), re.IGNORECASE)
        if len(list_matching) > 0:
            list_find_idx.append(1)
        else:
            list_find_idx.append(0)
    # The list where the target word exist (either as a whole word, or as a segment within longer word)
    list_word_idx = [i for i, x in enumerate(list_find_idx) if x == 1]
    list_word_len = [len(list_words[this_word_idx]) for this_word_idx in list_word_idx]
    if len(list_word_idx) == 0:
        print(f"The target word '{target_word}' doesn't exist.")
        return "error"
    elif len(list_word_idx) > 1:
        if len(list_word_len) != len(set(list_word_len)):
            print(f"Caution: there might exist more than one target word '{target_word}'? | {list_words}")
            return("error")
        # return "error"
    this_word_idx = list_word_idx[list_word_len.index(min(list_word_len))]
    return([this_word_idx])


def gen_all_pair_idx_list(list_words_1, list_words_2, target_word):
    idx_in_list_words_1 = get_idx_word_id(list_words_1, target_word)
    idx_in_list_words_2 = get_idx_word_id(list_words_2, target_word)
    # print(idx_in_list_words_1, idx_in_list_words_2)
    try:
        exclude_pair = [idx_in_list_words_1[0], idx_in_list_words_2[0]]
    except:
        print("Could not identify exclude_pair? The target word might not be present in the sentence, which is weird")
        exclude_pair = [np.nan, np.nan]
        # pdb.set_trace()
    list_all_pairs_idx = [pair for pair in list(itertools.product(range(len(list_words_1)), range(len(list_words_2)))) if pair != tuple(exclude_pair)]
    # print(len(list_all_pairs_idx))
    return(list_all_pairs_idx)


def get_idx_zh_word(this_tokenizer, this_sent, target_word, model_name='', is_debug=False):
    """
    Given a tokenizer, sentence and target word, return a dictionary that contains the information of how the tokenizer
    parse the sentence, and identify the target word.

    Parameter
    ---------
    this_tokenizer: Tokenizer
        Tokenizer from the transformers library
    this_sent: str
        Target sentence
    target_word: str
        Target word
    model_name: None (default)
        If provided, the dictionary returned would store the model_name
    
    Example
    -------
    ```python
    >>> from transformers import AutoTokenizer
    >>> model_name = "meta-llama/Llama-3.2-1B"
    >>> this_tokenizer = AutoTokenizer.from_pretrained(model_name)
    >>> target_word = "一线"
    >>> this_sent = "医护人员始终战斗在抗击疫情的一线。"
    >>> this_encode = this_tokenizer.encode_plus(this_sent, return_tensors="pt")
    >>> this_word_dict = get_idx_chinese_word(this_tokenizer, this_sent, target_word, model_name)
    >>> # The index this_word_dict['idx_start_char1'], this_word_dict['idx_end_char2'] can be used to
    >>> # slice the hidden representations
    """
    dict_sent = {} # This is a dictionary to store the find index results
    # char1 = target_word[0]
    # char2 = target_word[1]
    dict_sent["word"] = target_word
    dict_sent["sentence"] = this_sent
    # dict_sent["char1"] = char1
    # dict_sent["char2"] = char2

    if not model_name == None:
        dict_sent["model"] = model_name.split('/')[-1]
    # first of all, let's confirm the target word exists in the sentence
    if len(re.findall(target_word, this_sent)) < 1:
        print(f"ERROR | no {target_word} in : {this_sent}")
        dict_sent["error"] = 1
        # list_dict_sent.append(dict_sent)
        # list_error_dict_sent.append(dict_sent)
        # return(dict_sent)

    list_token_ids = this_tokenizer.encode(this_sent)
    try:
        list_decoded_tokens = [this_tokenizer.decode(this_token_id) for this_token_id in list_token_ids]
        stop_sliding = 0
        for len_window in range(1, 15):
            for i in range(len(list_token_ids)): # Changed
                this_combined = this_tokenizer.decode(list_token_ids[i:i+len_window])
                if len(re.findall('bert-base-chinese', model_name)) == 1:
                    this_combined = this_combined.replace(" ", "")
                if len(re.findall('bert-base-multilingual-uncased', model_name)) == 1:
                    this_combined = this_combined.replace(" ", "")
                # print(this_decoded)
                # this_combined = "".join(list_decoded_tokens[i:i+len_window])
                # print(f"{i} | {list_decoded_tokens[i:i+len_window]} | {this_combined} | {len(re.findall(target_word, this_combined))}")
                if len(re.findall(target_word, this_combined)) == 1:
                    idx_start_char1 = i
                    idx_end_char2 = i + (len_window-1)
                    # print(idx_start_char1, idx_end_char2)
                    stop_sliding = 1
                    break
            if stop_sliding == 1:
                break
        decoded_target_word = this_tokenizer.decode(list_token_ids[idx_start_char1:idx_end_char2+1])
        if len(re.findall('bert-base-chinese', model_name)) or len(re.findall('bert-base-multilingual-uncased', model_name)) == 1:
            decoded_target_word = decoded_target_word.replace(" ", "")
        
        if len(decoded_target_word) != 2:
            if is_debug:
                print(f"NOT MATCH: {this_sent} | {list_decoded_tokens} | {decoded_target_word}")
            dict_sent["match"] = 0
        else:
            if is_debug:
                print(f"{this_sent} | {list_decoded_tokens} | {decoded_target_word}")
            dict_sent["match"] = 1

        dict_sent["error"] = 0
        dict_sent["decoded_target_word"] = decoded_target_word
        dict_sent["idx_start_char1"] = idx_start_char1
        dict_sent["idx_end_char2"] = idx_end_char2
        dict_sent["idx_start_token"] = idx_start_char1
        dict_sent["idx_end_token"] = idx_end_char2
    except: 
        dict_sent["error"] = 1
    return(dict_sent)

def get_list_words_zh_jieba(this_sent: str) -> list:
    """
    Use jieba to segment the Chinese sentence and return a list of words

    Return
    ------
    A list of word: List
    """
    return(jieba.lcut(this_sent))


@debugging(False)
def gen_all_words_pairs(list_words_1, list_words_2, target_word) -> list:
    """Given two list of words and a target word, return all pairs of word excluding the pair that both items contain the target word
    
    Language-agnostic

    """
    list_all_word_pairs = list(itertools.product(list_words_1, list_words_2))
    try:
        for idx_word_pair, word_pair in enumerate(list_all_word_pairs):
            if (target_word in word_pair[0]) and (target_word in word_pair[1]):
                idx_exclude_pair = idx_word_pair
                print_debug(word_pair[0])
                print_debug(word_pair[1])
        list_all_word_pairs.pop(idx_exclude_pair)
    except:
        return []
    return(list_all_word_pairs)

def compute_cross_sent_word_similarity_zh(this_tokenizer, input_sent_1, input_sent_2, hidden_states_1, hidden_states_2, target_word, n_iter=50) -> pd.DataFrame:
    list_dict = []
    list_words_1 = get_list_words_zh_jieba(input_sent_1)
    list_words_2 = get_list_words_zh_jieba(input_sent_2)

    this_dict = {
        "sent1": input_sent_1,
        "sent2": input_sent_2,
        "target_word": target_word
    }
    try:    
        list_all_word_pairs = gen_all_words_pairs(list_words_1, list_words_2, target_word)
    except:
        return(this_dict)

    random.seed(20241010)
    random.shuffle(list_all_word_pairs)
    for word_pair in list_all_word_pairs[0:n_iter]:
        try:
            dict_word_1 = get_idx_zh_word(this_tokenizer, input_sent_1, word_pair[0])
            dict_word_2 = get_idx_zh_word(this_tokenizer, input_sent_2, word_pair[1])
            idx_word_tokens_1 = np.arange(dict_word_1['idx_start_char1'], dict_word_1['idx_end_char2']+1)
            idx_word_tokens_2 = np.arange(dict_word_2['idx_start_char1'], dict_word_2['idx_end_char2']+1)
            
            # print(idx_word_tokens_1, idx_word_tokens_2)

            hidden_states_word_1 = hidden_states_1[:, 0, :, :][:, idx_word_tokens_1, :].mean(axis=1)
            hidden_states_word_2 = hidden_states_2[:, 0, :, :][:, idx_word_tokens_2, :].mean(axis=1)

            torch_arr_1 = torch.from_numpy(hidden_states_word_1)
            torch_arr_2 = torch.from_numpy(hidden_states_word_2)
            cos_sim = CosineSimilarity(dim=1)
            this_cos_sim = cos_sim(torch_arr_1, torch_arr_2)
            for i in range(this_cos_sim.shape[0]):
                this_dict = {
                    "sent1": input_sent_1,
                    "sent2": input_sent_2,
                    "target_word": target_word,
                    "word1_idx_in_list_words_1": list_words_1.index(word_pair[0]),
                    "word2_idx_in_list_words_2": list_words_2.index(word_pair[1]),
                    "word1": word_pair[0],
                    "word2": word_pair[1],      
                    "layer": i,
                    "layer_rel": i/(this_cos_sim.shape[0]-1),
                    "cos_sim": this_cos_sim[i].item()
                }
                list_dict.append(this_dict)
        except:
            this_dict = {
                "sent1": input_sent_1,
                "sent2": input_sent_2,
                "target_word": target_word,
                "word1_idx_in_list_words_1": list_words_1.index(word_pair[0]),
                "word2_idx_in_list_words_2": list_words_2.index(word_pair[1]),
                "word1": word_pair[0],
                "word2": word_pair[1],
            }
            list_dict.append(this_dict)
    return(pd.DataFrame(list_dict))

def compute_cross_sent_word_similarity_en(this_tokenizer, input_sent_1, input_sent_2, hidden_states_1, hidden_states_2, target_word, n_iter=50) -> pd.DataFrame:
    list_dict = []
    list_words_1 = get_list_words(this_tokenizer, input_sent_1)
    list_words_2 = get_list_words(this_tokenizer, input_sent_2)

    this_dict = {
        "sent1": input_sent_1,
        "sent2": input_sent_2,
        "target_word": target_word
    }

    try:
        list_all_word_pairs = gen_all_words_pairs(list_words_1, list_words_2, target_word)
    except:
        print(this_dict)
    random.seed(20241010)
    random.shuffle(list_all_word_pairs)
    for word_pair in list_all_word_pairs[0:n_iter]:
        # print(word_pair)
        try:
            idx_word_tokens_1 = get_idx_word_id(list_words_1, target_word)
            idx_word_tokens_2 = get_idx_word_id(list_words_2, target_word)
            hidden_states_word_1 = hidden_states_1[:, 0, :, :][:, idx_word_tokens_1, :].mean(axis=1)
            hidden_states_word_2 = hidden_states_2[:, 0, :, :][:, idx_word_tokens_2, :].mean(axis=1)

            torch_arr_1 = torch.from_numpy(hidden_states_word_1)
            torch_arr_2 = torch.from_numpy(hidden_states_word_2)
            cos_sim = CosineSimilarity(dim=1)
            this_cos_sim = cos_sim(torch_arr_1, torch_arr_2)
            for i in range(this_cos_sim.shape[0]):
                this_dict = {
                    "sent1": input_sent_1,
                    "sent2": input_sent_2,
                    "target_word": target_word,
                    "word1_idx_in_list_words_1": list_words_1.index(word_pair[0]),
                    "word2_idx_in_list_words_2": list_words_2.index(word_pair[1]),
                    "word1": word_pair[0],
                    "word2": word_pair[1],      
                    "layer": i,
                    "layer_rel": i/(this_cos_sim.shape[0]-1),
                    "cos_sim": this_cos_sim[i].item()
                }
                list_dict.append(this_dict)
        except:
            this_dict = {
            "sent1": input_sent_1,
            "sent2": input_sent_2,
            "target_word": target_word,
            "word1_idx_in_list_words_1": list_words_1.index(word_pair[0]),
            "word2_idx_in_list_words_2": list_words_2.index(word_pair[1]),
            "word1": word_pair[0],
            "word2": word_pair[1],
            }
        list_dict.append(this_dict)
    return(pd.DataFrame(list_dict))





def compute_word_cossim(this_model, this_tokenizer, lists_sent, target_word, lang='en', is_custom_model=False, model_name=None):
    """
    Given a target word, and a list or two lists of sentences that contain the target word, compute the cosine similarity between the word representations
    
    """
    # Check how many list is provided
    n_list = len(lists_sent)
    match n_list:
        case 1:
            # Step 1: Compute the hidden representation first
            list_sent = lists_sent[0]
            n_sent = len(list_sent)
            list_representation = []
            for i in range(n_sent):
                # print(list_sent[i])
                try:
                    this_word_representation = get_word_representations(this_model, this_tokenizer, list_sent[i], target_word, lang=lang, is_custom_model=False, model_name=model_name)
                except:
                    # pdb.set_trace()
                    print(f"Errors while computing embedding for word: {target_word}")
                    return 0
                list_representation.append(this_word_representation)

            hidden_states = np.array(list_representation)
            # print(hidden_states.shape)

            # Step 2: Compute the cosine similarity
            from itertools import combinations
            idx_pairs = list(combinations(range(n_sent), 2)) # pair of index that doesn't include diagonal
            list_dict = []
            for idx_1, idx_2 in idx_pairs:
                cossim = CosineSimilarity(dim=1)
                this_pair_cossim = cossim(torch.from_numpy(hidden_states[idx_1, :, :]), torch.from_numpy(hidden_states[idx_2, :, :]))
                for idx_layer in range(this_pair_cossim.shape[0]):
                    this_dict = {
                        "idx_sent_1": idx_1,
                        "idx_sent_2": idx_2,
                        "sent1": list_sent[idx_1],
                        "sent2": list_sent[idx_2],
                        "layer": idx_layer,
                        "layer_rel": idx_layer/(this_pair_cossim.shape[0]-1),
                        "cos_sim": this_pair_cossim[idx_layer].item()
                    }
                    list_dict.append(this_dict)
        case 2:
            list_sent_1 = lists_sent[0]
            list_sent_2 = lists_sent[1]
            n_sent = len(list_sent_1)
            list_embedding_arr_1 = []
            list_embedding_arr_2 = []
            for i in range(n_sent):
                try:
                    this_embedding_arr_1 = get_word_representations(this_model, this_tokenizer, list_sent_1[i], target_word, lang=lang, is_custom_model=is_custom_model, model_name=model_name)
                    this_embedding_arr_2 = get_word_representations(this_model, this_tokenizer, list_sent_2[i], target_word, lang=lang, is_custom_model=is_custom_model, model_name=model_name)
                except:
                    # pdb.set_trace()
                    print(f"Errors while computing embedding for word: {target_word}")
                    return 0
                list_embedding_arr_1.append(this_embedding_arr_1)
                list_embedding_arr_2.append(this_embedding_arr_2)

            list_dict = []
            for this_comb in list(product(range(n_sent), range(n_sent))):
                try:
                    torch_arr_1 = torch.from_numpy(list_embedding_arr_1[this_comb[0]])
                    torch_arr_2 = torch.from_numpy(list_embedding_arr_2[this_comb[1]])
                    cos_sim = CosineSimilarity(dim=1)
                    this_cos_sim = cos_sim(torch_arr_1, torch_arr_2)
                    for i in range(this_cos_sim.shape[0]):
                        this_dict = {
                            "idx_sent_1": this_comb[0], # note that I intentionally make it 1-indexed for better readability
                            "idx_sent_2": this_comb[1], # same as above
                            "sent1": list_sent_1[this_comb[0]],
                            "sent2": list_sent_2[this_comb[1]],
                            "layer": i,
                            "layer_rel": i/(this_cos_sim.shape[0]-1),
                            "cos_sim": this_cos_sim[i].item()
                        }
                        list_dict.append(this_dict)
                except:
                    print(f"Some errors in computing cosine similarity for {target_word} | {list_sent_1[this_comb[0]]} | {list_sent_1[this_comb[1]]}")
                    continue
        case _:
            print("Length of variable lists_sent should be 1 or 2.")
    
    df_cos_sim = pd.DataFrame(list_dict).reset_index(drop=True)
    df_cos_sim['word'] = target_word
    df_cos_sim["cos_sim"] = df_cos_sim["cos_sim"].clip(-1, 1) # Sometimes there will be numerical precision error in which the cosine similarity is larger than one, so we do clipping here to avoid it.
    df_cos_sim["angle"] = np.arccos(df_cos_sim["cos_sim"]) * 180 / np.pi
    return(df_cos_sim)

from utils import get_list_words, get_idx_word_id, gen_all_pair_idx_list
def get_word_in_token_idx(this_tokenizer, this_sent, target_word):
    """
    Return the array of token indices that correspond to the target word within the provided sentence.

    Example
    ------- 
    >>> input_sent = "I am a neuroscientist studying neuroscience."
    >>> get_word_in_token_idx(this_tokenizer, input_sent, "neuroscientist")
    >>> array([4, 5, 6, 7, 8])
    """
    input_tokens = this_tokenizer.encode_plus(this_sent, return_tensors="pt")
    # input_tokens = input_tokens.to(device)
    list_words = get_list_words(this_tokenizer, this_sent)
    idx_word = get_idx_word_id(list_words, target_word)
    # Since idx_word is identified by assuming that list_words is exactly how the tokenizer determine the list of words
    # Cannot use regex to obtain the list of words.
    idx_word_tokens = np.where(np.array(input_tokens.word_ids()) == idx_word)[0]
    return(idx_word_tokens)


def get_token_idx_from_word_id(this_tokenizer, this_sent, target_word_id):
    """
    Given a corresponding word id, identify the token indices that encode this word.

    Note that we doesn't need to know what exactly the word is beforehand.

    Example
    -------
    >>> this_sent = "I have a lab with a research neuroscientist that Matthew's staff."
    >>> idx_word_tokens = get_token_idx_from_word_id(this_tokenizer, this_sent, 7)
    >>> print(idx_word_tokens)
    """
    this_encode = this_tokenizer.encode_plus(this_sent, return_tensors="pt")
    idx_word_tokens = np.where(np.array(this_encode.word_ids()) == target_word_id)[0]
    return(idx_word_tokens)

def decode_word_from_word_id(this_tokenizer, this_sent, target_word_id):
    """
    Given a corresponding word id, decode the word from the sentence.

    Example
    -------
    >>> this_sent = "I have a lab with a research neuroscientist that Matthew's staff."
    >>> decode_word_from_word_id(this_tokenizer, this_sent, 7)
    """
    this_encode = this_tokenizer.encode_plus(this_sent, return_tensors="pt")
    idx_word_tokens = get_token_idx_from_word_id(this_tokenizer, this_sent, target_word_id)
    return(this_tokenizer.decode(this_encode['input_ids'][0, idx_word_tokens]))

def compute_baseline_en(this_tokenizer, input_sent_1, input_sent_2, hidden_states_1, hidden_states_2, target_word, n_iter=30) -> pd.DataFrame:
    """
    Originally compute_cross_sent_word_similarity_en.

    Given two sentences, their corresponding hidden representations, target word and the tokenizer, compute the baseline cosine similarity. The baseline cosine similarity is defined as the averaged cosine similarity between any two random words drawn from the two sentence respectively. The pair containing two target words is excluded.
    
    Version: 20250118, to be uploaded to github, slightly lacked behin oif compute_baseline_zh
    """


    list_dict = []
    # list_word_tokens = []
    list_words_1 = get_list_words(this_tokenizer, input_sent_1)
    list_words_2 = get_list_words(this_tokenizer, input_sent_2)

    this_dict = {
        "sent1": input_sent_1,
        "sent2": input_sent_2,
        "target_word": target_word
    }

    try:
        # list_all_word_pairs = gen_all_words_pairs(list_words_1, list_words_2, target_word)
        list_all_word_pairs_idx = gen_all_pair_idx_list(list_words_1, list_words_2, target_word)
    except:
        print(this_dict)
    random.seed(20241010)
    random.shuffle(list_all_word_pairs_idx)
    # random.shuffle(list_all_word_pairs)
    match n_iter:
        case -1:
            n_iter = len(list_all_word_pairs_idx)
        case _:
            pass
    for word_id_1, word_id_2 in list_all_word_pairs_idx[0:n_iter]:
    # for word_pair in list_all_word_pairs[0:n_iter]:
        # print(word_pair)
        try:
            idx_word_tokens_1 = get_token_idx_from_word_id(this_tokenizer, input_sent_1, word_id_1)
            idx_word_tokens_2 = get_token_idx_from_word_id(this_tokenizer, input_sent_2, word_id_2)
            # idx_word_tokens_1 = get_word_in_token_idx(this_tokenizer, input_sent_1, word_pair[0])
            # idx_word_tokens_2 = get_word_in_token_idx(this_tokenizer, input_sent_2, word_pair[1])
            
            # list_word_tokens.append(tuple(idx_word_tokens_1, idx_word_tokens_2))

            # print("xxxxxxxxxxxxxx")
            # print(idx_word_tokens_1, idx_word_tokens_2)
            hidden_states_word_1 = hidden_states_1[:, 0, :, :][:, idx_word_tokens_1, :].mean(axis=1)
            hidden_states_word_2 = hidden_states_2[:, 0, :, :][:, idx_word_tokens_2, :].mean(axis=1)
            # print("--------------")

            torch_arr_1 = torch.from_numpy(hidden_states_word_1)
            torch_arr_2 = torch.from_numpy(hidden_states_word_2)
            cos_sim = CosineSimilarity(dim=1)
            this_cos_sim = cos_sim(torch_arr_1, torch_arr_2)
            for i in range(this_cos_sim.shape[0]):
                this_dict = {
                    "sent1": input_sent_1,
                    "sent2": input_sent_2,
                    "target_word": target_word,
                    "word1_idx_in_list_words_1": list_words_1.index(list_words_1[word_id_1]),
                    "word2_idx_in_list_words_2": list_words_2.index(list_words_2[word_id_2]),
                    "word1": list_words_1[word_id_1],
                    "word2": list_words_2[word_id_2],      
                    "layer": i,
                    "layer_rel": i/(this_cos_sim.shape[0]-1),
                    "cos_sim": this_cos_sim[i].item()
                }
                list_dict.append(this_dict)
        except:
            this_dict = {
            "sent1": input_sent_1,
            "sent2": input_sent_2,
            "target_word": target_word,
            "word1_idx_in_list_words_1": list_words_1.index(list_words_1[word_id_1]),
            "word2_idx_in_list_words_2": list_words_2.index(list_words_2[word_id_2]),
            "word1": list_words_1[word_id_1],
            "word2": list_words_2[word_id_2], 
            }
            list_dict.append(this_dict)
    this_df = pd.DataFrame(list_dict)
    this_df["cos_sim"] = this_df["cos_sim"].clip(-1, 1) # Sometimes there will be numerical precision error in which the cosine similarity is larger than one, so we do clipping here to avoid it.
    return(this_df)


def compute_baseline_zh(this_tokenizer, input_sent_1, input_sent_2, hidden_states_1, hidden_states_2, target_word, n_iter=30) -> pd.DataFrame:
    """
    Originally compute_cross_sent_word_similarity_zh.

    Given two sentences, their corresponding hidden representations, target word and the tokenizer, compute the baseline cosine similarity. The baseline cosine similarity is defined as the averaged cosine similarity between any two random words drawn from the two sentence respectively. The pair containing two target words is excluded.
    
    Version: 20250118, to be uploaded to github
    """
    list_dict = []
    list_words_1 = get_list_words_zh_jieba(input_sent_1)
    list_words_2 = get_list_words_zh_jieba(input_sent_2)

    this_dict = {
        "sent1": input_sent_1,
        "sent2": input_sent_2,
        "target_word": target_word
    }
    try:    
        list_all_word_pairs = gen_all_words_pairs(list_words_1, list_words_2, target_word)
    except:
        return(this_dict)

    random.seed(20241010)
    random.shuffle(list_all_word_pairs)
    count_iter = 0
    count_success = 0
    # while (count_success < n_iter) and (count_iter < len(list_all_word_pairs)):
    for word_pair in list_all_word_pairs:
        # if (count_success >= n_iter):
        if (count_iter >= n_iter):
            # If there is more than n_iter success, break the loop
            break
        # print(count_iter, count_success)
        # print(count_success < n_iter)
        # print(count_iter < len(list_all_word_pairs))
        try:
            dict_word_1 = get_idx_zh_word(this_tokenizer, input_sent_1, word_pair[0])
            dict_word_2 = get_idx_zh_word(this_tokenizer, input_sent_2, word_pair[1])
            idx_word_tokens_1 = np.arange(dict_word_1['idx_start_char1'], dict_word_1['idx_end_char2']+1)
            idx_word_tokens_2 = np.arange(dict_word_2['idx_start_char1'], dict_word_2['idx_end_char2']+1)

            hidden_states_word_1 = hidden_states_1[:, 0, :, :][:, idx_word_tokens_1, :].mean(axis=1)
            hidden_states_word_2 = hidden_states_2[:, 0, :, :][:, idx_word_tokens_2, :].mean(axis=1)

            torch_arr_1 = torch.from_numpy(hidden_states_word_1)
            torch_arr_2 = torch.from_numpy(hidden_states_word_2)
            cos_sim = CosineSimilarity(dim=1)
            this_cos_sim = cos_sim(torch_arr_1, torch_arr_2)
            for i in range(this_cos_sim.shape[0]):
                this_dict = {
                    "sent1": input_sent_1,
                    "sent2": input_sent_2,
                    "target_word": target_word,
                    "word1_idx_in_list_words_1": list_words_1.index(word_pair[0]),
                    "word2_idx_in_list_words_2": list_words_2.index(word_pair[1]),
                    "word1": word_pair[0],
                    "word2": word_pair[1],      
                    "layer": i,
                    "layer_rel": i/(this_cos_sim.shape[0]-1),
                    "cos_sim": this_cos_sim[i].item()
                }
                list_dict.append(this_dict)
            count_success += 1
        except:
            this_dict = {
                "sent1": input_sent_1,
                "sent2": input_sent_2,
                "target_word": target_word,
                "word1_idx_in_list_words_1": list_words_1.index(word_pair[0]),
                "word2_idx_in_list_words_2": list_words_2.index(word_pair[1]),
                "word1": word_pair[0],
                "word2": word_pair[1],
                "layer": np.nan,
                "layer_rel": np.nan,
                "cos_sim": np.nan
            }
            list_dict.append(this_dict)
        count_iter += 1
        # print(f"Iteration finished: {count_iter} | Succeed: {count_success}.")
    this_df = pd.DataFrame(list_dict)
    # this_df["cos_sim"] = this_df["cos_sim"].clip(-1, 1) # Sometimes there will be numerical precision error in which the cosine similarity is larger than one, so we do clipping here to avoid it.
    return(this_df)