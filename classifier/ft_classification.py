from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, cohen_kappa_score
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import os
import json
import time
import random
from rapidfuzz import fuzz
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from run_classifiers import group_duplicates, remove_duplicates, dcno_to_sentlab, gen_bn_sentlab, gen_mc_sentlab, classify_svm, res_dct_to_cls_rpt, cls_rpt_to_exp_rpt, encode_all_sents
import gc
from tqdm import tqdm
from datasets import DatasetDict

cwd = os.getcwd()
output_dir =  cwd+"/outputs/automodels_nofreeze"
input_dir =  cwd+"/inputs"
model_dir =  cwd+"/outputs/automodels_nofreeze"

# FUNCTION DEFINITIONS
# preprocessing

def ft_dsdct_classification(dsdct, int2label, label2int, model_address, cuda=False, batch=32):
    sentences = dsdct["holdout"]["text"]
    labels = dsdct["holdout"]["label"]
    num_lbs = len(set(labels))
    # load model
    dev = 'cuda' if cuda else None
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_address)
    print("Loading model")
    model = AutoModelForSequenceClassification.from_pretrained(model_address, num_labels=num_lbs,id2label=int2label, label2id=label2int).to(dev)
    model.eval()
    #for name, param in model.named_parameters():
    #    print(name, param.mean().item())
    print("Running model")
    prds_lst = []
    #with torch.no_grad():
    for i in tqdm(range(0, len(sentences), batch)):
        bsents = sentences[i : i + batch]
        test_embs = tokenizer(bsents, truncation=True, padding=True, return_tensors="pt").to(dev)
        logits = model(**test_embs).logits
        #prds = torch.argmax(logits, dim=1) #.cpu().numpy()
        prds = torch.max(logits,1).indices
        prds_lst.extend(prds.tolist())
        del test_embs, logits
        torch.cuda.empty_cache()
        gc.collect()
    raps = {'real': labels, 'pred': prds_lst}#[int2label[prd] for prd in prds_lst]}
    print("Freeing memory")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return raps

def run_experiments(int2label_dct, label2int_dct, cuda=False, scheck=False):
    models = {}
    for i in [3,6,9]:#range(10):
        models[f'bert_bn_e10_r{i}'] = model_dir+f'/paraphrase-xlm-r-multilingual-v1_bn_e10_r{i}.pt'
        models[f'bert_mc_e10_r{i}'] = model_dir+f'/paraphrase-xlm-r-multilingual-v1_mc_e10_r{i}.pt'
    print(list(models))
    results_dict = {'bn':{}, 'mc':{}}
    #
    #
    stw = time.time()
    for model in models:
        print(f'\nRunning model {model}')
        #print(torch.cuda.memory_allocated() / 1e9, "GB allocated before model")
        #print(torch.cuda.memory_reserved() / 1e9, "GB reserved before model")
        st = time.time()
        mode = model.split('_')[1]
        torch.cuda.empty_cache()
        if mode=='bn':
            r = model.split('_')[-1][1:]
            try:
                dsdct = DatasetDict.load_from_disk(output_dir+f"/{r}_bn")
                results_dict[mode][model] = ft_dsdct_classification(dsdct, int2label_dct[mode], label2int_dct[mode], models[model], cuda=cuda)
            except Exception as e:
                print(f"\nError in {model}: {e}\n")
        else:
            r = model.split('_')[-1][1:]
            try:
                dsdct = DatasetDict.load_from_disk(output_dir+f"/{r}_mc")
                results_dict[mode][model] = ft_dsdct_classification(dsdct, int2label_dct[mode], label2int_dct[mode], models[model], cuda=cuda)
            except Exception as e:
                print(f"\nError in {model}: {e}\n")
        print(f"{model} run completed in in {round(time.time()-st,2)}s")
        torch.cuda.empty_cache()
        gc.collect()
    etw = time.time()-stw
    print("Time elapsed total:", etw//60, "min and", round(etw%60), "sec")
    return results_dict

def main(outfn='30Feb', cuda = False, scheck = False):
    int2label_dct = {
        "bn": {
            0: "incentive",
            1: "non-incentive"
        },
        "mc":{
            0: "Fine",
            1: "Supplies",
            2: "Technical_assistance",
            3: "Tax_deduction",
            4: "Credit",
            5: "Direct_payment"
        }
    }
    label2int_dct = {
        "bn": {
            "incentive": 0,
            "non-incentive": 1
        },
        "mc":{
            "Fine": 0,
            "Supplies": 1,
            "Technical_assistance": 2,
            "Tax_deduction": 3,
            "Credit": 4,
            "Direct_payment": 5
        }
    }
    results_dict = run_experiments(int2label_dct, label2int_dct, cuda=cuda, scheck=scheck)
    with open(output_dir+f"/randp_{outfn}.json", 'w', encoding="utf-8") as outfile:
        json.dump(results_dict, outfile, ensure_ascii=False, indent=4)
    cls_rpt = res_dct_to_cls_rpt(results_dict, int2label_dct)
    exp_rpt = cls_rpt_to_exp_rpt(cls_rpt)
    with open(output_dir+f"/exprpt_{outfn}.json", 'w', encoding="utf-8") as outfile:
        json.dump(exp_rpt, outfile, ensure_ascii=False, indent=4)
    print('\nDone.')

if __name__ == "__main__":
    main(outfn='14Mar_automodel_nomods_evalin_test', cuda = True, scheck = False)
