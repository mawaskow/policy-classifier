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
from tqdm import tqdm
import torch
from rapidfuzz import fuzz
import numpy as np

cwd = os.getcwd()
output_dir =  cwd+"/outputs"
input_dir =  cwd+"/inputs"

# FUNCTION DEFINITIONS
# preprocessing

def group_duplicates(sents, labels, thresh = 90):
    '''
    Returns dictionary containing lists of sentence, label tuples in levenshtein groups.
    '''
    groups = []
    indices = set()
    # Group sentences by similarity
    for i, senti in enumerate(sents):
        # if i is already in indices, move on to next index
        if i in indices:
            continue
        new_group = [(senti, labels[i])]
        indices.add(i)
        for j, sentj in enumerate(sents):
            # only check sentences after current sentence [since prev sents will
            # already have been processed] and make sure sentence hasn't already
            # been added to another group [in indices]
            if j > i and j not in indices:
                lvnst = fuzz.ratio(senti, sentj)
                if lvnst >= thresh:
                    new_group.append((sentj, labels[j]))
                    indices.add(j)
        groups.append(new_group)
    print(f'{len(groups)} groups found with a threshold of {thresh}')
    # Convert groups to a dictionary with labels
    lvnst_grps = {}
    for i, group in enumerate(groups):
        lvnst_grps[f"group_{i}"] = group
    return lvnst_grps

def remove_duplicates(lvnst_grps):
    '''
    For dictionary of levenshtein groups, returns sentences, labels having
    converted each group into a single sentence, label entry.
    '''
    sents = []
    labels = []
    for group in lvnst_grps:
        sents.append(lvnst_grps[group][0][0])
        labels.append(lvnst_grps[group][0][1])
    print(f'Sanity check: {len(sents)} sentences and {len(labels)} labels')
    return sents, labels

def dcno_to_sentlab(dcno_json, sanity_check=False):
    '''
    For a json exported from doccano and read into a python dictionary,
    return the sentences and labels.
    '''
    sents = []
    labels = []
    for entry in dcno_json:
        if entry["label"] != []:
            if entry["label"][0].lower() !="unsure":
                sents.append(entry["text"])
                labels.append(entry["label"][0])
    print(f'Sanity Check: {len(sents)} sentences and {len(labels)} labels')
    if sanity_check:
        for i in range(2):
            n = random.randint(0, len(sents))
            print(f'[{n}] {labels[n]}: {sents[n]}')
    return sents, labels

# processing start

def gen_bn_sentlab(sents, labels, sanity_check=False):
    '''
    This gets the lists of the sentences for the binary classification: one list of incentives, one of non-incentives.
    inputs:
    sents - list of sentences
    labels - labels
    returns:
    inc - incentive sentences
    noninc - nonincentive sentences
    '''
    inc =[]
    noninc =[]
    for sent, label in zip(sents, labels):
        if label.lower() == "non-incentive":
            noninc.append(sent)
        else:
            inc.append(sent)
    i = len(inc)
    n = len(noninc)
    print(f'Sanity Check: {i} incentive sentences and {n} non-incentive sentences')
    print(f'Incentives: {i/(i+n)}; Non-Incentives: {n/(i+n)}')
    if sanity_check:
        n = random.randint(0, len(inc))
        print(f'[{n}] Incentive: {inc[n]}')
        n = random.randint(0, len(noninc))
        print(f'[{n}] Non-Incentive: {noninc[n]}')
    #return inc, noninc
    sentences = inc+noninc
    labels = ["incentive"]*len(inc)+["non-incentive"]*len(noninc)
    return sentences, labels

def gen_mc_sentlab(sents, labels, sanity_check=False):
    '''
    This fxn takes the list of sentences and the labels aggregated in the different methods
    and returns the incentive-specific sentences
    inputs:
    sents - list of sentences
    labels - labels
    outputs:
    sents - classified incentive sentences
    labs - classified incentive labels
    '''
    mc_sents = []
    mc_labels = []
    for sent, label in zip(sents, labels):
        if label.lower() == "non-incentive":
            continue
        else:
            mc_sents.append(sent)
            mc_labels.append(label)
    print(f'Sanity Check: {len(mc_sents)} incentive sentences and {len(mc_labels)} incentive labels')
    if sanity_check:
        for i in range(5):
            n = random.randint(0, len(mc_sents))
            print(f'[{n}] {mc_labels[n]}: {mc_sents[n]}')
    return mc_sents, mc_labels

def generate_dataset(sentences,labels,test_split=.2, r_state=9):
    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels, stratify=labels, test_size=test_split, random_state=r_state)
    finetuning = [{"text":text, "label":label} for text, label in zip(train_sents, train_labels)]
    testing = [{"text":text, "label":label} for text, label in zip(test_sents, test_labels)]
    ds = {"train":finetuning, "test":testing}
    return ds, f"_ts{int(test_split*100)}_r{r_state}"

# classification

def encode_all_sents(all_sents, sbert_model):
    '''
    modified from previous repository's latent_embeddings_classifier.py
    '''
    stacked = np.vstack([sbert_model.encode(sent) for sent in tqdm(all_sents)])
    return [torch.from_numpy(element).reshape((1, element.shape[0])) for element in stacked]

def classify_svm(train_embs, train_labels, test_embs, r_state= 9):
    print("Evaluating.")
    clf = svm.SVC(gamma=0.001, C=100., random_state=r_state)
    clf.fit(np.vstack(train_embs), train_labels)
    clf_preds = [clf.predict(sent_emb)[0] for sent_emb in test_embs]
    return clf_preds

def bn_classification(sentences, labels, cuda=False, r_state=9, exps=1, model_name= "paraphrase-xlm-r-multilingual-v1"):
    '''
    Takes incentive and nonincentive sentences, creates corresponding 
    label lists, and merges them accordingly. Splits data into trainig
    and testing sets, initializes a sentence transformer model,
    creates embeddings of the training and test sentences,
    returns encoded training sents, test sents, and test labels
    '''
    raps = {}
    #
    dev = 'cuda' if cuda else None
    print(f"Loading model {model_name}.")
    try:
        model = SentenceTransformer(model_name, device=dev) 
    except:
        model = SentenceTransformer(model_name, device=dev, trust_remote_code=True)
    for exp in range(exps):
        train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels, stratify=labels, test_size=0.2, random_state=exp)
        print("Encoding training sentences.")
        train_embs = encode_all_sents(train_sents, model)
        print("Encoding test sentences.")
        test_embs = encode_all_sents(test_sents, model)
        clf_prds = classify_svm(train_embs, train_labels, test_embs, r_state= r_state)
        raps[exp] = {'real': test_labels, 'pred': clf_prds}
    return raps

def mc_classification(sentences, labels, cuda=False, r_state=9, exps=1, model_name= "paraphrase-xlm-r-multilingual-v1", sanity_check=False):
    '''
    Takes sentences and labels. Splits data into trainig
    and testing sets, initializes a sentence transformer model,
    creates embeddings of the training and test sentences,
    returns encoded training sents, test sents, and test labels
    '''
    raps = {}
    # load model
    dev = 'cuda' if cuda else 'cpu'
    print(f"Loading model {model_name}.")
    try:
        model = SentenceTransformer(model_name, device=dev) 
    except:
        model = SentenceTransformer(model_name, device=dev, trust_remote_code=True)
    for exp in range(exps):
        train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels, stratify=labels, test_size=0.2, random_state=exp)
        label_names = list(set(train_labels))
        if sanity_check:
            print("Label names:", label_names)
        print("Encoding training sentences.")
        train_embs = encode_all_sents(train_sents, model)
        print("Encoding test sentences.")
        test_embs = encode_all_sents(test_sents, model)
        clf_prds = classify_svm(train_embs, train_labels, test_embs, r_state= r_state)
        raps[exp] = {'real': test_labels, 'pred': clf_prds}
    return raps

# evaluation

def res_dct_to_cls_rpt(res_dct, int2label_dct):
    '''
    Takes a real and predicted labels dictionary and returns
    dictionary of classification reports
    '''
    cls_rpt = {
        'bn':{},
        'mc':{}
    }
    for mode in list(res_dct):
        for model in list(res_dct[mode]):
            real = [int2label_dct[mode][res] for res in res_dct[mode][model]['real']]
            pred = [int2label_dct[mode][pred] for pred in res_dct[mode][model]['pred']]
            cls_rpt[mode][model] = classification_report(real, pred, output_dict=True)
    return cls_rpt

def cls_rpt_to_exp_rpt(cls_rpt):
    '''
    Takes dictionary of classification reports and returns dictionary of
    each classification model's average and sd values for accuracy and label
    f1-scores across the experiments.
    '''
    exp_rpt = {
        'bn':{},
        'mc':{}
    }
    for mode in list(cls_rpt):
        for model in list(cls_rpt[mode]):
            if mode == 'mc':
                exp_rpt[mode][model] = {
                    "accuracy": {},
                    "macroavg-f1": {},
                    "weightavg-f1": {},
                    "labels": {
                        'Credit':{},
                        'Direct_payment':{},
                        'Fine':{},
                        'Supplies':{},
                        'Tax_deduction':{},
                        'Technical_assistance':{}
                    }
                }
            else:
                exp_rpt[mode][model]= {
                    "accuracy": {},
                    "macroavg-f1": {},
                    "weightavg-f1": {},
                    "labels": {
                        'incentive':{},
                        'non-incentive':{}
                    }
                }
            accuracy = []
            macrof1 = []
            wghtf1=[]
            label_f1s = {}
            for label in list(exp_rpt[mode][model]["labels"]):
                label_f1s[label]=[]
            try:
                accuracy.append(cls_rpt[mode][model]['accuracy'])
                macrof1.append(cls_rpt[mode][model]['macro avg']["f1-score"])
                wghtf1.append(cls_rpt[mode][model]['weighted avg']["f1-score"])
            except:
                print(f'\nCould not add accuracy from {mode} {model}')
            for label in list(exp_rpt[mode][model]["labels"]):
                try:
                    label_f1s[label].append(cls_rpt[mode][model][label]["f1-score"])
                except:
                    print(f'\nCould not add F1 score for {label} in {mode} {model}')
            exp_rpt[mode][model]['accuracy'] = {'average':np.average(accuracy)}
            exp_rpt[mode][model]['macroavg-f1'] = {'average':np.average(macrof1)}
            exp_rpt[mode][model]['weightavg-f1'] = {'average':np.average(wghtf1)}
            for label in list(exp_rpt[mode][model]["labels"]):
                exp_rpt[mode][model]["labels"][label] = {'average':np.average(label_f1s[label])}
    return exp_rpt

def export_ds(sents, labels, fname):
    ds = []
    if len(sents) == len(labels):
        for i in range(len(sents)):
            ds.append({"text": sents[i], "label":labels[i]})
    with open(output_dir+f"/{fname}", 'w', encoding="utf-8") as outfile:
        json.dump(ds, outfile, ensure_ascii=False, indent=4)

def run_experiments(sentences, labels, exps=1, cuda=False, r_state=9, scheck=False):
    # experiments vary r state of train test split
    # r state for classifier is consistent
    models = {"sentence-transformers/paraphrase-xlm-r-multilingual-v1":'bert', "dunzhang/stella_en_1.5B_v5":'stella', "Alibaba-NLP/gte-Qwen2-1.5B-instruct":'qwen', "Alibaba-NLP/gte-large-en-v1.5":'glarg', "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":'minilm'}
    results_dict = {
        'bn':{
            'bert':{},
            'stella':{},
            'qwen':{},
            'glarg':{},
            'minilm':{}
        },
        'mc':{
            'bert':{},
            'stella':{},
            'qwen':{},
            'glarg':{},
            'minilm':{}
        }
    }
    #
    bn_sents, bn_labels = gen_bn_sentlab(sentences, labels, sanity_check=scheck)
    mc_sents, mc_labels = gen_mc_sentlab(sentences, labels, sanity_check=scheck)
    #export_ds(bn_sents, bn_labels, "bn_19Mar.json")
    #export_ds(mc_sents, mc_labels, "mc_19Mar.json")
    #print("\n\n\n\nYippee!\n\n\n\n")
    stw = time.time()
    for model in models:
        for mode in ['bn', 'mc']:
            if mode=='bn':
                results_dict[mode][models[model]] = bn_classification(bn_sents, bn_labels, r_state=r_state, cuda=cuda, exps=exps, model_name=model)
            else:
                results_dict[mode][models[model]] = mc_classification(mc_sents, mc_labels, r_state=r_state, cuda=cuda, exps=exps, model_name=model)
    etw = time.time()-stw
    print("Time elapsed total:", etw//60, "min and", round(etw%60), "sec")
    return results_dict

def main(sentences, labels, outfn='30Feb', cuda = False, exps=1, r_state=9, scheck = False):
    results_dict = run_experiments(sentences, labels, exps=exps, cuda=cuda, r_state=r_state, scheck=scheck)
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
    main(all_sents, all_labs, outfn='26Feb_ots_strat_3', exps=10, cuda = True, scheck = False)
