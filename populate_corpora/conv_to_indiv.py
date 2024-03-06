import json
import time
import argparse
import tqdm

def pre_tag_parse(input_path, outdir, experiment):
    '''
    Takes "Pre_tagged.json" generated in assisted_labelling.py
    resolves the queries and keys
    '''
    with open(input_path, "r", encoding="utf-8") as f:
        pretag = json.load(f)
    sentences = []
    labels = []
    for sentence in list(pretag):
        sentences.append(sentence)
        labels.append(pretag[sentence])
    with open(outdir+experiment+"_sentences.json", 'w') as f:
        json.dump(sentences, f, indent=2) 
    with open(outdir+experiment+"_labels.json", 'w') as f:
        json.dump(labels, f, indent=2) 

def hlt_parse(input_path, outdir, experiment):
    with open(input_path, "r", encoding="utf-8") as f:
        hlts = json.load(f)
    sentences=[]
    labels=[]
    for pdf in list(hlts):
        for pn in list(hlts[pdf]):
            for sn in list(hlts[pdf][pn]):
                sentences.append(hlts[pdf][pn][sn]["sentence"])
                labels.append(hlts[pdf][pn][sn]["label"])
    with open(outdir+experiment+"_sentences.json", 'w') as f:
        json.dump(sentences, f, indent=2) 
    with open(outdir+experiment+"_labels.json", 'w') as f:
        json.dump(labels, f, indent=2) 


def main():
    # from data augmentation
    pret_path = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\data_augmentation\\output\\EXP1\\Pre_tagged_0.5_fixed.json"
    #pret_path = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\data_augmentation\\output\\EXP1\\Pre_tagged_0.75_fixed.json"
    # from pdfs
    hlt_path = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\text_preprocessing\\output\\sents_join.json"
    #hlt_path = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\text_preprocessing\\output\\sents_split.json"
    #
    outdir = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\text_preprocessing\\output\\reduced\\"
    #pre_tag_parse(pret_path, outdir, experiment="Pre_tagged_0.5")
    hlt_parse(hlt_path, outdir, experiment="join")

if __name__ == '__main__':
    main()