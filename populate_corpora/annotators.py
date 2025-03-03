import random
import json
import os
from sklearn.metrics import cohen_kappa_score
from populate_corpora.data_cleaning import dcno_to_sentlab

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
    sents_c, labels_c = dcno_to_sentlab(dcno_json)
    label_lib = label_dct(dcno_json)
    resampled = resample_dict(label_lib)
    ann_frame = [{'text':sent, 'label':[]} for key in resampled.keys() for sent in resampled[key]]
    with open(cwd+"/inputs/subsample.json", 'w', encoding="utf-8") as outfile:
        json.dump(ann_frame, outfile, ensure_ascii=False, indent=4)
    # analyze
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

    s_sents, labels_sc, labels_sa = get_common_sentlabs(sents_c, labels_c, sentsa1, labelsa1)
    print(cohen_kappa_score(labels_sc, labels_sa))

    labs_binc = all_to_bin(labels_sc)
    labs_bina = all_to_bin(labels_sa)
    cohen_kappa_score(labs_binc, labs_bina)

    mclabsc, mclaba = all_to_sharedmc(labels_sc, labels_sa, labs_binc, labs_bina)
    print(len(mclabsc), len(mclaba))

    cohen_kappa_score(mclabsc, mclaba)