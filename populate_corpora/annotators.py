import random
import json
import os
from sklearn.metrics import cohen_kappa_score
from populate_corpora.data_cleaning import dcno_to_sentlab, group_duplicates, remove_duplicates, gen_mc_sentlab, gen_bn_lists
#from data_cleaning import dcno_to_sentlab, group_duplicates, remove_duplicates, gen_mc_sentlab, gen_bn_lists
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from collections import Counter

def label_dct(djson):
  rev_lib = {}
  for entry in djson:
        if entry["label"][0] not in rev_lib.keys():
          rev_lib[entry["label"][0]] = [entry["text"]]
        else:
          rev_lib[entry["label"][0]].append(entry["text"])
  return rev_lib

def resample_dict(label_lib):
  sampled_dct = {}
  labels = list(set(label_lib.keys()))
  labels.remove('Non-Incentive')
  for incentive in labels:
    if len(label_lib[incentive]) < 10:
      sampled_dct[incentive] = random.sample(label_lib[incentive], len(label_lib[incentive]))
    else:
      sampled_dct[incentive] = random.sample(label_lib[incentive], 10)
  sampled_dct['Non-Incentive'] = random.sample(label_lib['Non-Incentive'], 20)
  return sampled_dct 

def resample_forannot(sentences, labels, ts=0.1, ncs=0.6):
    '''
    ts is the ratio of the multiclass set we want to get a subsample of for annotation
    ncs is the ratio of the non-incentives to incentives we want
    '''
    print(Counter(labels))
    inc_sents, noninc_sents = gen_bn_lists(sentences, labels)
    mc_sents, mc_labels = gen_mc_sentlab(sentences, labels)
    tr_s, ann_sents, tr_l, ann_labels = train_test_split(mc_sents, mc_labels, stratify=mc_labels, test_size=ts, random_state=9)
    random.seed(9)
    ncf = ncs/(1-ncs)
    n_non = int(len(ann_sents)*ncf)
    ann_nonsents = random.sample(noninc_sents, n_non)
    ann_labels.extend(["Non-Incentive"]*n_non)
    ann_sents.extend(ann_nonsents)
    print(Counter(ann_labels))
    print("#sents==#labels:", len(ann_sents)==len(ann_labels))
    return ann_sents, ann_labels

def get_common_sentlabs(sents_c, labs_c, sents_a, labs_a):
    s_sents = []
    labels_sc = []
    labels_sa = []
    for ind_a, sent_a in enumerate(sents_a):
        if sent_a in sents_c:
            s_sents.append(sent_a)
            ind_c = sents_c.index(sent_a)
            labels_sc.append(labs_c[ind_c])
            labels_sa.append(labs_a[ind_a])
        else:
            continue
    return s_sents, labels_sc, labels_sa

def all_to_bin(all_labels):
    new = []
    for label in all_labels:
        if label == "Non-Incentive":
            new.append("Non-Incentive")
        else:
            new.append("Incentive")
    return new

def all_to_sharedmc(labs_c, labs_a, bin_labs_c, bin_labs_a):
    mclabsc, mclaba = [], []
    for i, labi in enumerate(bin_labs_c):
        if labi == "Incentive" and bin_labs_a[i] == "Incentive":
            mclabsc.append(labs_c[i])
            mclaba.append(labs_a[i])
    return mclabsc, mclaba

def main():
    cwd = os.getcwd()
    # generate
    with open(cwd+"/inputs/19Jan25_firstdatarev.json","r", encoding="utf-8") as f:
        dcno_json = json.load(f)
    with open(cwd+"/inputs/27Jan25_query_checked.json","r", encoding="utf-8") as f:
        qry_json = json.load(f)
    sents1, labels1 = dcno_to_sentlab(dcno_json)
    sents2, labels2 = dcno_to_sentlab(qry_json)
    # merge original and augmented datasets
    sents2.extend(sents1)
    labels2.extend(labels1)
    all_sents, all_labs = remove_duplicates(group_duplicates(sents2,labels2,thresh=90))
    ann_sents, ann_labels = resample_forannot(all_sents, all_labs, 0.35, 0.6)
    print(round(len(ann_sents)/len(all_sents), 3))
    ann_frame = [{'text':ann_sents[i], 'label':[]} for i in range(len(ann_labels))]
    with open(cwd+"/inputs/subsamplexxx.json", 'w', encoding="utf-8") as outfile:
        json.dump(ann_frame, outfile, ensure_ascii=False, indent=4)
    val_frame = [{'text':ann_sents[i], 'label':ann_labels[i]} for i in range(len(ann_labels))]
    with open(cwd+"/inputs/subsample_keyxxx.json", 'w', encoding="utf-8") as outfile:
        json.dump(val_frame, outfile, ensure_ascii=False, indent=4)
    print("Done")
    # analyze
    '''
    with open(cwd+"/inputs/annotation_odon.json","r", encoding="utf-8") as f:
        ann_json = json.load(f)
    sentsa, labelsa = dcno_to_sentlab(ann_json)
    # normalize labels
    sentsa1, labelsa1 = [], []
    swap_labs = {'non-incentive':'Non-Incentive', 'fine':'Fine', 'tax deduction':'Tax_deduction', 'credit':'Credit', 'direct payment':'Direct_payment', 'supplies':'Supplies', 'technical assistance':'Technical_assistance'}
    for i, lab in enumerate(labelsa):
        try:
            labelsa1.append(swap_labs[lab])
            sentsa1.append(sentsa[i])
        except:
            pass

    s_sents, labels_sc, labels_sa = get_common_sentlabs(all_sents, all_labs, sentsa1, labelsa1)
    print(cohen_kappa_score(labels_sc, labels_sa))

    labs_binc = all_to_bin(labels_sc)
    labs_bina = all_to_bin(labels_sa)
    cohen_kappa_score(labs_binc, labs_bina)

    mclabsc, mclaba = all_to_sharedmc(labels_sc, labels_sa, labs_binc, labs_bina)
    print(len(mclabsc), len(mclaba))

    cohen_kappa_score(mclabsc, mclaba)
    '''

if __name__ == "__main__":
   main()