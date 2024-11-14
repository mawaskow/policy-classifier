'''
This script is based off of the original repository's jupyter notebook
'''

import spacy
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import time
import cupy as cp
import json

from latent_embeddings_classifier import *
from utils import *

def classify_bn(incentives, nonincentives, rs = None):
    '''
    eventually going to split this data_load into subfunctions
    but at the moment we are just trying to adapt this from the jupiter notebook
    '''

    incent_lbls = ["incentive"]*len(incentives)
    noninc_lbls = ["non-incentve"]*len(nonincentives)

    sentences = incentives+nonincentives
    labels = incent_lbls+noninc_lbls
    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels, test_size=0.2, random_state=rs)
    print("Loading model.")
    bin_model = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")
    print("Encoding training sentences.")
    all_sent_embs = encode_all_sents(train_sents, bin_model)
    # they noted that the projection matrix made stuff worse
    #proj_matrix = cp.asnumpy(calc_proj_matrix(train_sents, 50, es_nlp, bin_model, 0.01))
    #all_sent_embs = encode_all_sents(test_sents, bin_model, proj_matrix)
    print("Encoding test sentences.")
    all_test_embs = encode_all_sents(test_sents, bin_model)
    print("Evaluating.")
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=9)
    clf.fit(np.vstack(all_sent_embs), train_labels)
    clf_preds = [clf.predict(sent_emb)[0] for sent_emb in all_test_embs]
    report_bn = classification_report(test_labels, clf_preds, output_dict=False)
    return report_bn
    
def classify_mc(sentences, labels, rs=None):
    '''
    eventually going to split this into subfunctions
    but at the moment we are just trying to adapt this from the jupiter notebook
    '''
    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels, test_size=0.2, random_state=rs)
    # eventually will be able to load pretrained model from training via a path?
    print("Loading model.")
    bin_model = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")
    print("Encoding training sentences.")
    all_sent_embs = encode_all_sents(train_sents, bin_model)

    # they noted that the projection matrix made stuff worse
    #proj_matrix = cp.asnumpy(calc_proj_matrix(train_sents, 50, es_nlp, bin_model, 0.01))
    #all_sent_embs = encode_all_sents(test_sents, bin_model, proj_matrix)
    print("Encoding test sentences.")
    all_test_embs = encode_all_sents(test_sents, bin_model)
    print("Evaluating.")
    # classifier 1
    '''
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=9)
    clf.fit(np.vstack(all_sent_embs), train_labels)
    clf_preds = [clf.predict(sent_emb)[0] for sent_emb in all_test_embs]
    rfc_rpt = classification_report(test_labels, clf_preds)
    print(rfc_rpt)
    '''
    #logger.info(rfc_rpt)
    # classifier 2
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(np.vstack(all_sent_embs), train_labels)
    clf_preds = [clf.predict(sent_emb)[0] for sent_emb in all_test_embs]
    svc_rpt = classification_report(test_labels, clf_preds)
    #logger.info(svc_rpt)
    # last classifiers
    #additional_analysis(all_sent_embs, train_labels, all_test_embs, test_labels)
    return svc_rpt

######################

def additional_analysis(train_sent_embs, train_labels, test_sent_embs, test_labels):
    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier(n_estimators=2000,
                        feature_fraction=0.06,
                        bagging_fraction=0.67,
                        bagging_freq=1,
                        verbose=0,
                        n_jobs=6,
                        random_state=69420)

    gb_classifiers = [lgbm] #, gbm, xgb, ab #cb
    gb_names = [i.__class__.__name__ for i in gb_classifiers]

    for clf, clf_name in zip(gb_classifiers, gb_names):
        print("Evaluating:", clf_name)
        print("Training...")
        clf.fit(np.vstack(train_sent_embs), train_labels)
        print("Predicting...")
        clf_preds = [clf.predict(sent_emb)[0] for sent_emb in test_sent_embs]
        lgbm_rpt = classification_report(test_labels, clf_preds)
        print(lgbm_rpt)
        #logger.info(lgbm_rpt)

#############################

def label_conversion(labels):
  '''
  This fxn goes through the list of labels to create two additional aggregations of the data.
  In the "class-heavy"/"cheavy" lists, labels containing "mention" become the class itself (removing the "mention-")
  The "bnry" lists have only "non-incentive"/"incentive" whereas
  the others have "non-incentive"/each named class.
  input:
  labels - list of labels
  returns:
  labels_bnry_cheavy - binary labels, class heavy
  labels_classhvy - multiclass labels, class heavy
  '''
  labels_mc_c = []
  labels_bn_c = []

  # binary
  for label in labels:
    if label == "non-incentive":
      labels_bn_c.append("non-incentive")
    elif label[0:7] == "mention":
      labels_bn_c.append("incentive")
    else:
      labels_bn_c.append("incentive")

  print("Labels, binary:", len(labels_bn_c))
  # multiclass
  for label in labels:
    if label == "non-incentive":
      labels_mc_c.append("non-incentive")
    elif label[0:7] == "mention":
      labels_mc_c.append(label[8:])
    else:
      labels_mc_c.append(label)

  print("Labels, incentive-class:", len(labels_mc_c))
  print(set(labels_mc_c))

  return labels_bn_c, labels_mc_c

def bn_sent_list_gen(sents, labels_bn_c):
  '''
  This gets the lists of the sentences for the binary classification: one list of incentives, one of non-incentives.
  inputs:
  sents - list of sentences
  labels_classhvy - binary labels class heavy
  returns:
  inc_nonhvy - incentive sentences, non-incentive heavy
  noninc_nonhvy - nonincentive sentences, non-incentive heavy
  inc_clshvy - incentive sentences, class heavy
  noninc_clshvy - nonincentive sentences, class heavy
  '''
  inc_c =[]
  noninc_c =[]

  for sent, label in zip(sents, labels_bn_c):
    if label == "non-incentive":
      noninc_c.append(sent)
    else:
      inc_c.append(sent)
  return inc_c, noninc_c

def mc_data_prep(sents, labels_mc_c):
  '''
  This fxn takes the list of sentences and the labels aggregated in the different methods
  and returns the incentive-specific sentences
  inputs:
  sents - list of sentences
  labels_classhvy - labels multiclass but class heavy
  outputs:
  sents_clshvy - classified incentive sentences, class heavy
  nlabs_clshvy - classified incentive labels, class heavy
  '''
  sents_c = []
  labs_c = []

  for sent, label in zip(sents, labels_mc_c):
    if label == "non-incentive":
      pass
    else:
      sents_c.append(sent)
      labs_c.append(label)

  print("Incentive sents from class heavy agg: ", len(sents_c), len(labs_c))
  return sents_c, labs_c

########################################

def main():
    st = time.time()
    if spacy.prefer_gpu():
        print("Using the GPU")
    else:
        print("Using the CPU")

    input_path = "C:/Users/Allie/Documents/GitHub/policy-classifier/populate_corpora/outputs/"

    with open(os.path.join(input_path,"pret_sents.json"), "r", encoding="utf-8") as f:
        sim_sent = json.load(f)
    with open(os.path.join(input_path,"neg_bin_sents.json"), "r", encoding="utf-8") as f:
        dissim_sent = json.load(f)

    data_load(sim_sent, dissim_sent)

    et = time.time()-st
    print("Time elapsed total:", et//60, "min and", round(et%60), "sec")

if __name__ == '__main__':
    main()