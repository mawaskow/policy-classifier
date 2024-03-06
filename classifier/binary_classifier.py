import spacy
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import time
import cupy as cp
import json

from data_loading.src.utils import *
#from model_evaluation.src.model_evaluator import *
from data_augmentation.src.zero_shot_classification.latent_embeddings_classifier import *

def data_load(incentv_path, nonincv_path):
    '''
    eventually going to split this into subfunctions
    but at the moment we are just trying to adapt this from the jupiter notebook
    '''
    with open(incentv_path, "r", encoding="utf-8") as f:
        incentives = json.load(f)
    with open(nonincv_path, "r", encoding="utf-8") as f:
        nonincentives = json.load(f)

    incent_lbls = ["incentive"]*len(incentives)
    noninc_lbls = ["non-incentve"]*len(nonincentives)

    sentences = incentives+nonincentives
    labels = incent_lbls+noninc_lbls
    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels, test_size=0.2)
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
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=9)
    clf.fit(np.vstack(all_sent_embs), train_labels)
    clf_preds = [clf.predict(sent_emb)[0] for sent_emb in all_test_embs]
    print(classification_report(test_labels, clf_preds))

def main():
    st = time.time()
    if spacy.prefer_gpu():
        print("Using the GPU")
    else:
        print("Using the CPU")
    # load spacy model
    #es_nlp = spacy.load('es_core_news_lg')

    sim_sent = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\data_augmentation\\output\\EXP1\\Pre_tagged_0.5_sentences.json"
    dissim_sent = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\data_augmentation\\output\\EXP1\\Pre_tagged_dis_sentences.json"

    data_load(sim_sent, dissim_sent)
    et = time.time()-st
    print("Time elapsed total:", et//60, "min and", round(et%60), "sec")

if __name__ == '__main__':
    main()