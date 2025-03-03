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

def all_to_bin(all_labels):
    new = []
    for label in all_labels:
        if label == "Non-Incentive":
            new.append("Non-Incentive")
        else:
            new.append("Incentive")
    return new

def main():
    cwd = os.getcwd()
    with open(cwd+"/inputs/19Jan25_firstdatarev.json","r", encoding="utf-8") as f:
        dcno_json = json.load(f)
    label_lib = label_dct(dcno_json)
    resampled = resample_dict(label_lib)
    ann_frame = [{'text':sent, 'label':[]} for key in resampled.keys() for sent in resampled[key]]
    with open(cwd+"/inputs/subsample.json", 'w', encoding="utf-8") as outfile:
        json.dump(ann_frame, outfile, ensure_ascii=False, indent=4)
    
    with open(cwd+"/inputs/annotation_odon.json","r", encoding="utf-8") as f:
        ann_json = json.load(f)

    sentsa, labelsa = dcno_to_sentlab(ann_json)
    sentsa1, labelsa1 = [], []
    swap_labs = {'non-incentive':'Non-Incentive', 'fine':'Fine', 'tax deduction':'Tax_deduction', 'credit':'Credit', 'direct payment':'Direct_payment', 'supplies':'Supplies', 'technical assistance':'Technical_assistance'}
    for i, lab in enumerate(labelsa):
        try:
            labelsa1.append(swap_labs[lab])
            sentsa1.append(sentsa[i])
        except:
            pass

    print(len(sentsa1), len(labelsa1))

    labelsc = []
    s_sents = []
    labelsa2 = []

    for inda, sent in enumerate(sentsa1):
        if sent in sents1:
            s_sents.append(sent)
            indc = sents1.index(sent)
            labelsc.append(labels1[indc])
            labelsa2.append(labelsa1[inda])
        else:
            continue
    for i in [labelsc, s_sents, labelsa2]:
        print(len(i))

    print(cohen_kappa_score(labelsc, labelsa2))

    binlabsc = all_to_bin(labelsc)
    binlaba = all_to_bin(labelsa2)

    cohen_kappa_score(binlabsc, binlaba)

    mclabsc, mclaba = [], []
    for i, labi in enumerate(binlabsc):
        if labi == "Incentive" and binlaba[i] == "Incentive":
            mclabsc.append(labelsc[i])
            mclaba.append(labelsa2[i])
    print(len(mclabsc), len(mclaba))

    cohen_kappa_score(mclabsc, mclaba)