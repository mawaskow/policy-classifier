'''
This script is based off of the original repository's jupyter notebook
'''

import spacy
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import time
#import cupy as cp
import json
import logging
logger = logging.getLogger(__name__)

from utils import *
from latent_embeddings_classifier import *

def data_load(sentences, labels):
    '''
    eventually going to split this into subfunctions
    but at the moment we are just trying to adapt this from the jupiter notebook
    '''
    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels, test_size=0.2)
    
    label_names = unique_labels(train_labels)
    print("Label names:", label_names)
    print("Train Sentence:", train_sents[2], "\nTrain Label:", train_labels[2])
    print("Test Sentence:", test_sents[2], "\nTest Label:", test_labels[2])
    
    logger.info("Label names:", label_names)
    logger.info("Train Sentence:", train_sents[2], "\nTrain Label:", train_labels[2])
    logger.info("Test Sentence:", test_sents[2], "\nTest Label:", test_labels[2])
    # these are how to evaluate your data distributions
    #label_names = unique_labels(train_labels)
    #numeric_train_labels = labels2numeric(train_labels, label_names)
    #plot_data_distribution(numeric_train_labels, label_names)
    #
    # load model
    model_name = "paraphrase-xlm-r-multilingual-v1"
    # eventually will be able to load pretrained model from training via a path?
    print("Loading model.")
    bin_model = SentenceTransformer(model_name)
    print("Encoding training sentences.")
    all_sent_embs = encode_all_sents(train_sents, bin_model)
    
    # they noted that the projection matrix made stuff worse
    #proj_matrix = cp.asnumpy(calc_proj_matrix(train_sents, 50, es_nlp, bin_model, 0.01))
    #all_sent_embs = encode_all_sents(test_sents, bin_model, proj_matrix)
    print("Encoding test sentences.")
    all_test_embs = encode_all_sents(test_sents, bin_model)
    print("Evaluating.")
    # classifier 1
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=9)
    clf.fit(np.vstack(all_sent_embs), train_labels)
    clf_preds = [clf.predict(sent_emb)[0] for sent_emb in all_test_embs]
    rfc_rpt = classification_report(test_labels, clf_preds)
    print(rfc_rpt)
    logger.info(rfc_rpt)
    # classifier 2
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(np.vstack(all_sent_embs), train_labels)
    clf_preds = [clf.predict(sent_emb)[0] for sent_emb in all_test_embs]
    svc_rpt = classification_report(test_labels, clf_preds)
    print(svc_rpt)
    logger.info(svc_rpt)
    # last classifiers
    additional_analysis(all_sent_embs, train_labels, all_test_embs, test_labels)

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
        logger.info(lgbm_rpt)

def main():
    logging.basicConfig(filename='../outputs/myapp.log', level=logging.INFO)
    logger.info('Started')
    st = time.time()
    if spacy.prefer_gpu():
        print("Using the GPU")
    else:
        print("Using the CPU")
    # load spacy model
    #es_nlp = spacy.load('es_core_news_lg')

    #input_path = "C:/Users/Allie/Documents/GitHub/policy-classifier/populate_corpora/outputs/"
    input_path = "../inputs/ChatGPT"

    with open(os.path.join(input_path,"sents_noact.json"), "r", encoding="utf-8") as f:
        sents = json.load(f)
    with open(os.path.join(input_path,"labs_noact.json"), "r", encoding="utf-8") as f:
        labels = json.load(f)

    data_load(sents, labels)

    et = time.time()-st
    print("Time elapsed total:", et//60, "min and", round(et%60), "sec")

if __name__ == '__main__':
    main()