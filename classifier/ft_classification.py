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
from latent_embeddings_classifier import encode_all_sents
import numpy as np

from run_classifiers import group_duplicates, remove_duplicates, dcno_to_sentlab, gen_bn_sentlab, gen_mc_sentlab, classify_svm, res_dct_to_cls_rpt, cls_rpt_to_exp_rpt

cwd = os.getcwd()
output_dir =  cwd+"/outputs/models"
input_dir =  cwd+"/inputs"

# FUNCTION DEFINITIONS
# preprocessing

def ft_classification(sentences, labels, model_address, cuda=False, exps=1, r=9):
    raps = {}
    # load model
    dev = 'cuda' if cuda else None
    #tts_r =int(model_address.split("_")[-1].split('.')[0])
    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels,stratify=labels, test_size=0.2, random_state=r)
    print(f"Loading model {model_address.split('/')[-1]}.")
    model = SentenceTransformer(model_address, device=dev)
    print("Encoding training sentences.")
    train_embs = encode_all_sents(train_sents, model)
    print("Encoding test sentences.")
    test_embs = encode_all_sents(test_sents, model) 
    for exp in range(exps):
        clf_prds = classify_svm(train_embs, train_labels, test_embs, r_state= exp)
        raps[exp] = {'real': test_labels, 'pred': clf_prds}
    return raps

def run_experiments(sentences, labels, exps=1, cuda=False, scheck=False, r=9):
    # r should be same r used in traintestsplit
    models = {}
    for i in range(10):
        models[f'bert_bn_e10_r{i}'] = output_dir+f'/paraphrase-xlm-r-multilingual-v1_bn_epochs_10_rstate_{i}.pt'
        models[f'bert_mc_e10_r{i}'] = output_dir+f'/paraphrase-xlm-r-multilingual-v1_mc_epochs_10_rstate_{i}.pt'
    print(list(models))
    results_dict = {'bn':{}, 'mc':{}}
    #
    bn_sents, bn_labels = gen_bn_sentlab(sentences, labels, sanity_check=scheck)
    mc_sents, mc_labels = gen_mc_sentlab(sentences, labels, sanity_check=scheck)
    bn_ft_sents, bn_ho_sents, bn_ft_labels, bn_ho_labels = train_test_split(bn_sents, bn_labels, stratify=bn_labels, test_size=0.2, random_state=r)
    mc_ft_sents, mc_ho_sents, mc_ft_labels, mc_ho_labels = train_test_split(mc_sents, mc_labels, stratify=mc_labels, test_size=0.25, random_state=r)
    stw = time.time()
    for model in models:
        print(f'Running model {model}')
        mode = model.split('_')[1]
        if mode=='bn':
            try:
                results_dict[mode][model] = ft_classification(bn_ho_sents, bn_ho_labels, models[model], cuda=cuda, exps=exps, r=r)
            except:
                pass
        else:
            try:
                results_dict[mode][model] = ft_classification(mc_ho_sents, mc_ho_labels, models[model], cuda=cuda, exps=exps, r=r)
            except:
                pass
    etw = time.time()-stw
    print("Time elapsed total:", etw//60, "min and", round(etw%60), "sec")
    return results_dict

def main(sentences, labels, outfn='30Feb', cuda = False, exps=1, scheck = False, r=9):
    results_dict = run_experiments(sentences, labels, exps=exps, cuda=cuda, scheck=scheck)
    with open(output_dir+f"/randp_{outfn}.json", 'w', encoding="utf-8") as outfile:
        json.dump(results_dict, outfile, ensure_ascii=False, indent=4)
    cls_rpt = res_dct_to_cls_rpt(results_dict)
    exp_rpt = cls_rpt_to_exp_rpt(cls_rpt)
    with open(output_dir+f"/exprpt_{outfn}.json", 'w', encoding="utf-8") as outfile:
        json.dump(exp_rpt, outfile, ensure_ascii=False, indent=4)
    print('\nDone.')

if __name__ == "__main__":
    with open(input_dir+"/19Jan25_firstdatarev.json","r", encoding="utf-8") as f:
        dcno_json = json.load(f)
    with open(input_dir+"/27Jan25_query_checked.json","r", encoding="utf-8") as f:
        qry_json = json.load(f)
    sents1, labels1 = dcno_to_sentlab(dcno_json)
    sents2, labels2 = dcno_to_sentlab(qry_json)
    # merge original and augmented datasets
    sents2.extend(sents1)
    labels2.extend(labels1)
    all_sents, all_labs = remove_duplicates(group_duplicates(sents1,labels1,thresh=90))
    main(all_sents, all_labs, outfn='28Feb_ft_berts_4', exps=10, cuda = True, scheck = False, r=69)
