import torch
#from transformers import GPT2Tokenizer, GPT2Model
from transformers import GPT2Tokenizer, GPTJModel
#from sklearn.manifold import TSNE
import numpy as np
from datasets import load_dataset
import pandas as pd

from os import listdir
from os.path import isfile, join

import pyarrow as pa
import pyarrow.parquet as pq

pd.options.display.max_columns = 100

from dotenv import load_dotenv
import os

load_dotenv()

def average_reps(subtokens, bert_reps, tokens):
    ""
    final_reps = []
    sub_reps = []
    subtok = ""
    i = 0
    for j, (tok, rep) in enumerate(zip(subtokens, bert_reps)):
        print(tok)
        if tok == "Ġ" or tok == 'ĉ' :
            continue
        if "Ã¸" in tok:
            tok = tok.replace('Ã¸','ø')
        if "Ã¥" in tok:
            tok = tok.replace('Ã¥','å')
        if "Ãħ" in tok:
            tok = tok.replace('Ãħ','Å')
        if "Ã¦" in tok:
            tok = tok.replace('Ã¦','æ')
        if "Â«" in tok:
            tok = tok.replace('Â«','«')
        if "Â»" in tok:
            tok = tok.replace('Â»','»')
        if "Ãĺ" in tok:
            tok = tok.replace('Ãĺ','Ø')
        if "Ã¡" in tok:
            tok = tok.replace('Ã¡','á')
        if "Ãł" in tok:
            tok = tok.replace('Ãł','à')
        if "Ã³" in tok:
            tok = tok.replace('Ã³','ó')
        if "ÃŃ" in tok:
            tok = tok.replace('ÃŃ','í')
        if "Ã©" in tok:
            tok = tok.replace('Ã©','é')
        if "ĠâĢĶ" in tok:
            tok = tok.replace('ĠâĢĶ','—')
        if "ĠâĢĵ" in tok:
            tok = tok.replace('ĠâĢĵ','–')
        if "ĠâĢĿ" in tok:
            tok = tok.replace('ĠâĢĿ','”')
        # if "Ċ" in tok:
        #     tok = tok.replace('Ċ', '\n')
        if "Ġ" in tok:
            subtok += tok[1:]
        else:
            subtok += tok
        sub_reps.append(rep.detach().numpy())
        if "âĢĻ" in subtok:
            subtok = subtok.replace('âĢĻ','’')
        if "âĢľ" in subtok:
            subtok = subtok.replace('âĢľ','“')
        if "Ã¼" in subtok:
            subtok = subtok.replace('Ã¼','ü')
        if subtok == tokens[i]:
            ave_rep = np.array(sub_reps).mean(axis=0)
            final_reps.append(ave_rep)
            sub_reps = []
            print(subtok)
            print(tokens[i])
            subtok = ""
            i += 1
        elif tok == "[UNK]":
            # Have to account for some cases where the tokenizer breaks a noizy
            # token up and doesn't add ##, such as '”A' -> ["[UNK]", "A"]
            if j < len(subtokens) - 1:
                next_subtok = subtokens[j + 1]
                if next_subtok in tokens[i]:
                    #subtok = tokens[i].replace(next_subtok, "")
                    #subtok = tokens[i][:1]
                    subtok = subtok[:-5]
                    subtok += '”'
                    sub_reps.append(rep.detach().numpy())
                else:
                    ave_rep = np.array(sub_reps).mean(axis=0)
                    final_reps.append(ave_rep)
                    sub_reps = []
                    #print(subtok)
                    subtok = ""
                    i += 1
            else:
                ave_rep = np.array(sub_reps).mean(axis=0)
                final_reps.append(ave_rep)
                sub_reps = []
                #print(subtok)
                subtok = ""
                i += 1
    return np.array(final_reps)

def get_sample(file):
    out_file = file.rsplit('/',1)
    out_file = out_file[-1]
    out_file = out_file.rsplit('.',1)[0]
    out_sampled = out_file + "_sampled.csv"
    out_file = out_file + '_code_embed.csv'

    if isfile(out_sampled):
        print("Load data from existing file {}...".format(out_sampled))
        df_sampled = pd.read_csv(out_sampled)
    else:
        print("Reading samples from {}...".format(file))
        df_finetuned = pd.read_parquet(file)
        print("Number of original samples is {}.".format(len(df_finetuned)))
        df_sampled = df_finetuned.sample(frac=0.3)
        print("Number of sampled codes is {}.".format(len(df_sampled)))
        print("Saving sampled file to csv...")
        df_sampled.to_csv(out_sampled)

    return df_sampled, out_file

def get_embedding(input_text, tokenizer, model):
    input_text = input_text.strip()
    encoded_input = tokenizer(input_text, return_tensors='pt')
    output = model(**encoded_input)
    token_reps = output.last_hidden_state.squeeze(0)
    #subtokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])
    #print(">>>>>>>>>>>")
    #print(token_reps.shape)
    #print(">>>>>>>>>>>")
    #print(subtokens)
    t_ave_reps = torch.mean(token_reps, 0)
    #t_ave_reps = t_ave_reps.detach().numpy()
    t_max_reps = torch.max(token_reps, 0)
    t_max_reps = t_max_reps.values

    t_ave_reps = t_ave_reps.detach().numpy()
    t_max_reps = t_max_reps.detach().numpy()
    t_ave_reps = ','.join(map(str, t_ave_reps))
    t_max_reps = ','.join(map(str, t_max_reps))
    #print(type(t_ave_reps))
    #print(type(t_max_reps))

    return t_ave_reps, t_max_reps


def main():
    # load GPT2 tokenizer and GPTJ pre-trianed model
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-j-6B',use_auth_token=os.environ['HF_TOKEN'])
    model = GPTJModel.from_pretrained('andstor/gpt-j-6B-smart-contract', use_auth_token=os.environ['HF_TOKEN'])
    model.eval()
    mypath = "comment_only/finetuned/"
    files = [join(mypath, f) for f in listdir(mypath)]
    print(files)
    
    sample_ratio = 0.2
    for file in files:
        df_sampled, out_file = get_sample(file)

        for index, row in df_sampled.iterrows():
            if index % 100 == 0:
                print(index)
            
            g_code, o_code = row["generated"], row["func_code"]
            if not g_code.strip() or not o_code.strip():
                print("Empty code detected!")
                continue

            try:
                # get generated code embeddings
                g_ave_reps, g_max_reps = get_embedding(g_code, tokenizer, model)
                # get original code embeddings
                o_ave_reps, o_max_reps = get_embedding(o_code, tokenizer, model)

                # Convert row series to a new dataframe
                df = row.to_frame().T
                
                df['g_ave_reps'] = g_ave_reps
                df['g_max_reps'] = g_max_reps
                df['o_ave_reps'] = o_ave_reps
                df['o_max_reps'] = o_max_reps
                #df['g_ave_reps'] = [g_ave_reps.detach().numpy()] * len(df)
                #df['g_max_reps'] = [g_max_reps] * len(df)
                #df['o_ave_reps'] = [o_ave_reps.detach().numpy()] * len(df)
                #df['o_max_reps'] = [o_max_reps] * len(df)
                
            except Exception as e:
                print(e)
                continue

            #print('Dumping new row to file...')
            if isfile(out_file):
                df.to_csv(out_file, mode='a', header=False)
            else:
                df.to_csv(out_file)

    print("DONE!")
 

if __name__ == "__main__":
    main()
    '''
    #test
    fname = "generated_comments_0_10000_slack_256_code_embed.csv"
    df = pd.read_csv(fname)
    s = df['g_ave_reps'][0]
    l = s.strip().split(',')
    print(len(l))
    print(type(df['g_ave_reps'][0]))
    #print(df.head())
    '''
    
    




