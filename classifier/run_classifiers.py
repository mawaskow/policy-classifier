import spacy
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
    if sanity_check:
        print(f'Sanity Check: {len(sents)} sentences and {len(labels)} labels')
        for i in range(2):
            n = random.randint(0, len(sents))
            print(f'[{n}] {labels[n]}: {sents[n]}')
    return sents, labels

# processing start

def gen_bn_lists(sents, labels, sanity_check=False):
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
    if sanity_check:
        i = len(inc)
        n = len(noninc)
        print(f'Sanity Check: {i} incentive sentences and {n} non-incentive sentences')
        print(f'Incentives: {i/(i+n)}; Non-Incentives: {n/(i+n)}')
        n = random.randint(0, len(inc))
        print(f'[{n}] Incentive: {inc[n]}')
        n = random.randint(0, len(noninc))
        print(f'[{n}] Non-Incentive: {noninc[n]}')
    return inc, noninc

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
    if sanity_check:
        print(f'Sanity Check: {len(mc_sents)} incentive sentences and {len(mc_labels)} incentive labels')
        for i in range(5):
            n = random.randint(0, len(mc_sents))
            print(f'[{n}] {mc_labels[n]}: {mc_sents[n]}')
    return mc_sents, mc_labels

def bn_generate_embeddings(incentives, nonincentives, cuda=False, r_state=9, model_name= "paraphrase-xlm-r-multilingual-v1", sanity_check=False):
    '''
    Takes incentive and nonincentive sentences, creates corresponding 
    label lists, and merges them accordingly. Splits data into trainig
    and testing sets, initializes a sentence transformer model,
    creates embeddings of the training and test sentences,
    returns encoded training sents, test sents, and test labels
    '''
    incent_lbls = ["incentive"]*len(incentives)
    noninc_lbls = ["non-incentive"]*len(nonincentives)
    sentences = incentives+nonincentives
    labels = incent_lbls+noninc_lbls
    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels, test_size=0.2, random_state=r_state)
    #
    print("Loading model.")
    if cuda:
        bin_model = SentenceTransformer(model_name, device="cuda") # or .cuda()
    else:
        bin_model = SentenceTransformer(model_name)
    print("Encoding training sentences.")
    train_embs = encode_all_sents(train_sents, bin_model)
    print("Encoding test sentences.")
    test_embs = encode_all_sents(test_sents, bin_model)
    if sanity_check:
        n = random.randint(0, len(train_sents))
        print(f'[{n}]: {train_embs[n]}')
        t = random.randint(0, len(test_sents))
        print(f'{t}: {test_embs[t]}')
    return train_embs, test_embs, train_labels, test_labels

def mc_generate_embeddings(sentences, labels, cuda=False, r_state=9, model_name= "paraphrase-xlm-r-multilingual-v1", sanity_check=False):
    '''
    Takes sentences and labels. Splits data into trainig
    and testing sets, initializes a sentence transformer model,
    creates embeddings of the training and test sentences,
    returns encoded training sents, test sents, and test labels
    '''
    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels, test_size=0.2, random_state=r_state)
    label_names = list(set(train_labels))
    if sanity_check:
        print("Label names:", label_names)
        n = random.randint(0, len(train_labels))
        print(f"[{n}] {train_labels[n]}: {train_sents[n]}")
        t = random.randint(0, len(test_labels))
        print(f"[{t}] {test_labels[t]}: {test_sents[t]}")
    # load model
    print("Loading model.")
    if cuda:
        bin_model = SentenceTransformer(model_name, device="cuda") # or .cuda()
    else:
        bin_model = SentenceTransformer(model_name)
    print("Encoding training sentences.")
    train_embs = encode_all_sents(train_sents, bin_model)
    print("Encoding test sentences.")
    test_embs = encode_all_sents(test_sents, bin_model)
    if sanity_check:
        n = random.randint(0, len(train_sents))
        print(f'[{n}]: {train_embs[n]}')
        n = random.randint(0, len(test_sents))
        print(f'{n}: {test_embs[n]}')
    return train_embs, test_embs, train_labels, test_labels

def classify_rf(train_embs, train_labels, test_embs, r_state= 9):
    print("Evaluating.")
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=r_state)
    clf.fit(np.vstack(train_embs), train_labels)
    clf_preds = [clf.predict(sent_emb)[0] for sent_emb in test_embs]
    return clf_preds

def classify_svm(train_embs, train_labels, test_embs, r_state= 9):
    print("Evaluating.")
    clf = svm.SVC(gamma=0.001, C=100., random_state=r_state)
    clf.fit(np.vstack(train_embs), train_labels)
    clf_preds = [clf.predict(sent_emb)[0] for sent_emb in test_embs]
    return clf_preds

def main(cuda = False, scheck = False):
    with open(input_dir+"/19Jan25_firstdatarev.json","r", encoding="utf-8") as f:
        dcno_json = json.load(f)
    with open(input_dir+"/27Jan25_query_checked.json","r", encoding="utf-8") as f:
        qry_json = json.load(f)
    sents1, labels1 = dcno_to_sentlab(dcno_json, sanity_check=scheck)
    sents2, labels2 = dcno_to_sentlab(qry_json, sanity_check=scheck)
    # merge original and augmented datasets
    sents2.extend(sents1)
    labels2.extend(labels1)
    all_sents, all_labs = remove_duplicates(group_duplicates(sents2,labels2,thresh=90))
    inc_sents, noninc_sents = gen_bn_lists(all_sents, all_labs, sanity_check=scheck)
    mc_sents, mc_labels = gen_mc_sentlab(all_sents, all_labs, sanity_check=scheck)
    #
    results_dict = {
        'bn':{
            'bert':{},
            'stella':{},
            'qwen':{}
        },
        'mc':{
            'bert':{},
            'stella':{},
            'qwen':{}
        }
    }
    stw = time.time()
    bn_train_embs, bn_test_embs, bn_train_labels, bn_test_labels = bn_generate_embeddings(inc_sents, noninc_sents, cuda, r_state=9, sanity_check=scheck)
    bn_rf_pred = classify_rf(bn_train_embs, bn_train_labels, bn_test_embs)
    bn_svm_pred = classify_svm(bn_train_embs, bn_train_labels, bn_test_embs)
    results_dict['bn']['bert']['rf'] = classification_report(bn_test_labels, bn_rf_pred, output_dict=True)
    results_dict['bn']['bert']['svm'] = classification_report(bn_test_labels, bn_svm_pred, output_dict=True)
    mc_train_embs, mc_test_embs, mc_train_labels, mc_test_labels = mc_generate_embeddings(mc_sents, mc_labels, cuda, r_state=9, sanity_check=scheck)
    mc_rf_pred = classify_rf(mc_train_embs, mc_train_labels, mc_test_embs)
    mc_svm_pred = classify_svm(mc_train_embs, mc_train_labels, mc_test_embs)
    results_dict['mc']['bert']['rf'] = classification_report(mc_test_labels, mc_rf_pred, output_dict=True)
    results_dict['mc']['bert']['svm'] = classification_report(mc_test_labels, mc_svm_pred, output_dict=True)
    etw = time.time()-stw
    print("Time elapsed total:", etw//60, "min and", round(etw%60), "sec")
    with open(output_dir+"/results_xxx.json", 'w', encoding="utf-8") as outfile:
        json.dump(results_dict, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main(cuda = True, scheck = True)