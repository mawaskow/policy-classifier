import spacy
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, cohen_kappa_score
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import time
import os
import json
#import cupy as cp
import random
#R = random.Random(9)
#random.seed(9)
import datetime

from binary_classifier import classify_bn, classify_mc, label_conversion, bn_sent_list_gen, mc_data_prep
from utils import *
'''
Functions
'''

def prep_data(input_path):
    with open(input_path,"r", encoding="utf-8") as f:
        dcno_json = json.load(f)
    sents = []
    labels = []
    for entry in dcno_json:
        if entry["label"] != []:
            if entry["label"][0].lower() !="unsure":
                sents.append(entry["text"])
                labels.append(entry["label"][0])
    return sents, labels

def main():
    seed=9
    cwd = os.getcwd()
    input_path = cwd+"/inputs/admin.json"
    output_dir =  cwd+f"/outputs/{round(datetime.datetime.now().timestamp())}.txt"
    f = open(output_dir, "w")
    f.write(str(datetime.datetime.now()))

    sents, labels = prep_data(input_path)
    labels_bn, labels_mc = label_conversion(labels)
    incentives, nonincentives = bn_sent_list_gen(sents, labels_bn)

    #sents, labels = mc_data_prep(sents, labels_mc)
    sents_inc, labels_inc = mc_data_prep(sents, labels_mc)
    print('Sanity Check:')
    print(len(sents), len(labels))
    print(len(labels_bn), len(labels_mc))
    print(len(labels_inc), len(labels_inc))
    print("Incentives to Non-Incentives")
    print(len(incentives), len(nonincentives))
    
    report_binary = classify_bn(incentives, nonincentives, rs=seed) 
    report_multiclass = classify_mc(sents_inc, labels_inc, rs=seed)

    f.write(report_binary)
    f.write(report_multiclass)
    f.close()

    print(report_binary)
    print(report_multiclass)

if __name__ == '__main__':
    main()