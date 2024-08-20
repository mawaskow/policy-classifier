# -*- coding: utf-8 -*-
"""
Based on original repo's /tasks/text_preprocessing/src/sentence_split_local.py which would clean and split the pdf texts into sentences as well
as format them for processing outputted a json file of the pdf names and their (partially cleaned) extracted text.
>sents_json = {file_id: {"metadata":
>                {"n_sentences": len(postprocessed_sents),
>                "language": language},
>                "sentences": postprocessed_sents}}
>with open(os.path.join(output_dir, f'{file_id}_sents.json'), 'w') as f:
The old version would save the sentences in different jsons for every file.
This would be helpful for manual labelling.
We should keep it as an option but not the only one.

In this version: 
Clean PDF texts, split into sentences, and format for training
Clean PDF annotations-- both labels and highlights, format for training, then option to split sentences
[format data augmentation things for training?]

###############

First add cleaning of full texts, then add cleaning of highlights, and cleaning of labels

Eventually adapt so that language is not just spanish

Bring back saving into independent sentence files for each pdf (for manual labeling)

"""
import json
import os
from pathlib import Path
import argparse
from tqdm import tqdm
#
from typing import Dict, List, Any, Set
import os
import nltk
#nltk.download('punkt')
import json
import argparse
import unidecode
from utils import *

def text_cleaning(text):
    """
    From previous repository
    Cleans a piece of text by removing escaped characters.
    Args:
        text (str): string with text
    Returns:
        str: cleaned piece of text
    """
    # Remove escaped characters
    escapes = ''.join([chr(char) for char in range(1, 32)])
    text = text.translate(str.maketrans('', '', escapes))
    return text

##########################################################################################
#   First handle full texts
##########################################################################################

def preprocess_text(txt: str, remove_new_lines: bool = False) -> str:
    """
    From previous repository
    Steps in the preprocessing of text:
        0. Run text cleaning script (moved from pdf to json script)
        1. Remove HTML tags
        2. Replace URLS by a tag [URL]
        3. Replace new lines and tabs by normal spaces - sometimes sentences have new lines in the middle
        4. Remove excessive spaces (more than 1 occurrence)
        5. Parse emails and abreviations
    """
    txt = text_cleaning(txt)
    txt = replace_links(remove_html_tags(txt)).strip()
    if remove_new_lines:
        txt = txt.replace("\n", " ").replace("\t", " ").strip()
    txt = remove_multiple_spaces(txt)
    txt = parse_emails(txt)
    txt = parse_acronyms(txt)
    new_txt = ""
    all_period_idx = set([indices.start() for indices in re.finditer("\.", txt)])
    for i, char in enumerate(txt):
        if i in all_period_idx:
            # Any char following a period that is NOT a space means that we should not add that period
            if i + 1 < len(txt) and txt[i + 1] != " ":
                continue
            # NOTE: Any char that is a number following a period will not count.
            # For enumerations, we're counting on docs being enumerated as "(a)" or "(ii)", and if not,
            # they will be separated by the "." after the number:
            # "Before bullet point. 3. Bullet point text" will just be "Before bullet point 3." and "Bullet point text" as the sentences
            if i + 2 < len(txt) and txt[i + 2].isnumeric():
                continue
            # If we wanted to have all numbered lists together, uncomment this, and comment out the previous condition
            # if i + 2 < len(txt) and not txt[i + 2].isalpha():
            #     continue
        new_txt += char
    return new_txt

def preprocess_english_text(txt: str, remove_new_lines: bool = False) -> str:
    '''
    From previous repository
    '''
    return preprocess_text(txt, remove_new_lines)

def preprocess_spanish_text(txt: str, remove_new_lines: bool = False) -> str:
    '''
    From previous repository
    '''
    return unidecode.unidecode(preprocess_text(txt, remove_new_lines))

def remove_short_sents(sents: List[str], min_num_words: int = 4) -> List[str]:
    """
    From previous repository
    Remove sentences that are made of less than a given number of words. Default is 4
    """
    return [sent for sent in sents if len(sent.split()) >= min_num_words]

def get_nltk_sents(txt: str, tokenizer: nltk.PunktSentenceTokenizer, extra_abbreviations: Set[str] = None) -> List[str]:
    '''
    From previous repository
    '''
    if extra_abbreviations is not None:
        tokenizer._params.abbrev_types.update(extra_abbreviations)
    return tokenizer.tokenize(txt)

def format_sents_for_output(sents, doc_id):
    """
    Fxn from Firebanks-Quevedo repo
    """
    formatted_sents = {}

    for i, sent in enumerate(sents):
        formatted_sents.update({f"{doc_id}_sent_{i}": {"text": sent, "label": []}})

    return formatted_sents

def format_sents_for_new_output(sents, doc_id):
    formatted_sents = {}

    for i, sent in enumerate(sents):
        formatted_sents.update({f"{doc_id}_sent_{i}": {"text": sent, 
                                                       "info": 
                                                            {"label":[],
                                                             "type":
                                                             {
                                                                 "action": [],
                                                                 "class": []
                                                             }}
                                                       }})

    return formatted_sents

def get_clean_text_dct(pdf_conv, tokenizer):
    '''
    Takes a dictionary of full text of pdf files and returns all sentences, cleaned, in one list
    Input:
        pdf_conv (dct): dictionary of full text of pdf files
    Output: 
        Error files
    Returns:
        sentences (lst): all sentences, cleaned
    '''
    language='english'
    abbrevs= None
    min_num_words = 5
    file_lst = []
    for key in pdf_conv:
        file_lst.append((key,pdf_conv[key]['Text']))
    error_files={}
    i = 0
    folder_dct = {}
    for file_id, text in tqdm(file_lst):
        try:
            preprocessed_text = preprocess_english_text(text)
            sents = get_nltk_sents(preprocessed_text, tokenizer, abbrevs)
            postprocessed_sents = format_sents_for_output(remove_short_sents(sents, min_num_words), file_id)
            folder_dct[file_id] = postprocessed_sents
        except Exception as e:
            error_files[str(file_id)]= str(e)
        i += 1
    print(f'Number of error files: {len(error_files)}')
    return folder_dct

def get_clean_new_text_dct(pdf_conv, tokenizer):
    '''
    Takes a dictionary of full text of pdf files and returns all sentences, cleaned, in one list
    Input:
        pdf_conv (dct): dictionary of full text of pdf files
    Output: 
        Error files
    Returns:
        sentences (lst): all sentences, cleaned
    '''
    language='english'
    abbrevs= None
    min_num_words = 5
    file_lst = []
    for key in pdf_conv:
        file_lst.append((key,pdf_conv[key]['text']))
    error_files={}
    i = 0
    folder_dct = {}
    for file_id, text in tqdm(file_lst):
        try:
            preprocessed_text = preprocess_english_text(text)
            sents = get_nltk_sents(preprocessed_text, tokenizer, abbrevs)
            postprocessed_sents = format_sents_for_new_output(remove_short_sents(sents, min_num_words), file_id)
            folder_dct[file_id] = postprocessed_sents
        except Exception as e:
            error_files[str(file_id)]= str(e)
        i += 1
    print(f'Number of error files: {len(error_files)}')
    return folder_dct

def get_clean_text_sents(pdf_conv):
    '''
    Takes a dictionary of full text of pdf files and returns all sentences, cleaned, in one list
    Input:
        pdf_conv (dct): dictionary of full text of pdf files
    Output: 
        Error files
    Returns:
        sentences (lst): all sentences, cleaned
    '''
    language='spanish'
    abbrevs= None
    tokenizer = ES_TOKENIZER
    min_num_words = 5
    sentences = []
    file_lst = []
    for key in pdf_conv:
        file_lst.append((key,pdf_conv[key]['Text']))
    error_files={}
    i = 0
    for file_id, text in tqdm(file_lst):
        try:
            preprocessed_text = preprocess_spanish_text(text)
            sents = get_nltk_sents(preprocessed_text, tokenizer, abbrevs)
            postprocessed_sents = remove_short_sents(sents, min_num_words)
            sentences+=postprocessed_sents
        except Exception as e:
            error_files[str(file_id)]= str(e)
        i += 1
    print(f'Number of error files: {len(error_files)}')
    return sentences

def format_fulltxt_for_json(pdf_conv):
    """
    """
    formatted_sents = []
    language='english'
    txts = []
    file_lst = []
    for key in pdf_conv:
        file_lst.append((key,pdf_conv[key]['Text']))
    error_files={}
    i = 0
    for file_id, text in tqdm(file_lst):
        try:
            preprocessed_text = preprocess_english_text(text)
            txts.append(preprocessed_text)
        except Exception as e:
            error_files[str(file_id)]= str(e)
        i += 1
    print(f'Number of error files: {len(error_files)}')

    for i, txt in enumerate(txts):
        formatted_sents.append({"text": txt, "label": []})

    return formatted_sents

##########################################################################################
#   Clean annotations
##########################################################################################
# first clean labels
#################################################
# initialize incentive classification lists and dictionaries
ann_cls_lst = [
    "Direct payment",
    "Credit",
    "Tax deduction",
    "Technical assistance",
    "Supplies",
    "Fine"
]

ann_cls_dct = {
    "direct": "Direct payment",
    "direct payment": "Direct payment",
    "credit": "Credit",
    "tax deduction": "Tax deduction",
    "technical assistance": "Technical assistance",
    "supplies": "Supplies",
    "fine": "Fine",
    "techical assistance": "Technical assistance",
    "tax credit": "Tax deduction",
    "pes": "Direct payment",
    "technical support": "Technical assistance",
    "direct payments": "Direct payment",
    "fines":"Fine"
}

def merge_label(input_string):
    '''
    input: label (str)
    returns: cleaned label (list)
    Goes through the raw label and converts it to
    elements a list if it can be converted into
    a known label
    '''
    label_lst = []
    try:
        input_string = input_string.split(",")
        for i in input_string:
            i = i.strip().lower()
            if i in ann_cls_dct.keys():
                label_lst.append(ann_cls_dct[i])
            elif i[:15] in ann_cls_dct.keys():
                label_lst.append(ann_cls_dct[i[:15]])
            else:
                i=i.replace("(", "")
                i=i.replace(")","")
                i=i.split(" ")
                for j in i:
                    if j=='pes':
                        label_lst.append("Direct payment")
        return label_lst
    except Exception as e:
        return [e]

def label_show_str(input_path):
    '''
    input: file path (str) to json
    output: prints foreign label,
            prints #foreign labels out of #labels
    Shows the labels in a file that dont match the known labels
    when the file labels are strings (not list elements)
    '''
    with open(input_path, "r", encoding="utf-8") as f:
        pdf_ann = json.load(f)
    i=0
    j=0
    #traverse by filename
    for fn in pdf_ann.keys():
        # then pagenumber
        for pn in pdf_ann[fn].keys():
            #then sentence in page
            for sn in pdf_ann[fn][pn].keys():
                j+=1
                label = pdf_ann[fn][pn][sn]['label']
                if label not in ann_cls_lst:
                    print(label)
                    i=i+1
    print(i, j)

def label_show_lst(ann_dct):
    '''
    #####################
    come back to this one and make it adaptable for any label?
    #####################
    input: file path (str) to json
    output: prints foreign label,
            prints #foreign labels out of #labels
    Shows the labels in a file that arent known labels
    when the file labels are list elements
    '''
    i=0
    j=0
    #traverse by filename
    for fn in ann_dct.keys():
        # then pagenumber
        for pn in ann_dct[fn].keys():
            #then sentence in page
            for sn in ann_dct[fn][pn].keys():
                j+=1
                label = ann_dct[fn][pn][sn]['label']
                if label!=[]:
                    print(label)
                for lb in range(len(label)):
                    if label[lb] not in ann_cls_lst:
                        print(label[lb])
                        i=i+1
    print(f"Errors in {i}/{j} labels")

def clean_labels(annot_dct):
    '''
    inputs: annotation dct
    output: file with clean/uniform labels
    Takes json file with raw annotations and converts them into lists of uniform labels (in new file)
    '''
    #label_show_str(input)
    '''
    # testing labels
    input_strings = ["Direct payment (PES)", "Forest, Agriculture (Crop)", "Direct payment (PES), Credit,",
                     "Credit, Technical assistance", " a policy in itself but a very important criticism of the inadequacies of previous financial incentive programs, namely PRODEFOR, PROCYMAF, PRODEPLAN. This argues that the programs lack long-term security due to dependency on budgets and lack of private investment. It also states that the previous programs were inflexible to different regional situations.",
                     "Fines", "Direct payments (PES), Technical assistance", "Other (Environmental education)",
                     "Unknown, Technical assistance", "PES, credit, technical assistance"]
    for i in input_strings:
        print(i, "BECOMES", merge_label(i))
    '''
    for pdf in tqdm(annot_dct.keys()):
        for pg in annot_dct[pdf].keys():
            for snt in annot_dct[pdf][pg].keys():
                # label cleaning
                label = text_cleaning(annot_dct[pdf][pg][snt]["label"]).split("\n")[0]
                label = label.split("\r")[0][3:]
                annot_dct[pdf][pg][snt]["label"] = merge_label(label)
    #label_show_lst(annot_dct)
    return annot_dct

def remove_empty_labels(empties):
    for pdf in list(empties):
        for pg in list(empties[pdf]):
            for snt in list(empties[pdf][pg]):
                if len(empties[pdf][pg][snt]["label"]) == 0:
                    empties[pdf][pg].pop(snt)
            if len(empties[pdf][pg].keys()) == 0:
                empties[pdf].pop(pg)
        if len(empties[pdf].keys()) == 0:
            empties.pop(pdf)
    return empties

##################################################
# next clean highlights
###################################################

def sentcheck_dups(pdf_ann):
    '''
    input: checks for duplicate texts in highlights
    output: 
    '''
    #traverse by filename
    for fn in list(pdf_ann):
        # then pagenumber
        for pn in list(pdf_ann[fn]):
            if len(list(pdf_ann[fn][pn]))>1:
                # create a new dictionary for the page
                new_pg = {}
                # traverse sentence keys by iterable
                for si in range(len(list(pdf_ann[fn][pn]))):
                    # if first sentence, add to new page (sentence and label)
                    if si== 0:
                        sn = list(pdf_ann[fn][pn])[si]
                        new_pg[sn] = pdf_ann[fn][pn][sn]
                    else:
                        # get previous and current sentence keys
                        pk = list(pdf_ann[fn][pn])[si-1]
                        ck = list(pdf_ann[fn][pn])[si]
                        # check if previous sentence is contained in present sentence
                        if pdf_ann[fn][pn][pk]["sentence"] in pdf_ann[fn][pn][ck]["sentence"]:
                            # if so, replace sentence and add label back
                            new_pg[ck] = {}
                            new_pg[ck]["sentence"] = pdf_ann[fn][pn][ck]["sentence"].replace(pdf_ann[fn][pn][pk]["sentence"], "")
                            new_pg[ck]["label"] = pdf_ann[fn][pn][ck]["label"]
                        else:
                            new_pg[ck] = pdf_ann[fn][pn][ck]
                pdf_ann[fn][pn] = new_pg
    return pdf_ann

def keep_paragraph(hlt_dct):
    '''
    keeps highlights in their paragraph shapes but cleans them
    '''
    abbrevs= None
    tokenizer = ES_TOKENIZER
    min_num_words = 5
    for pdf in tqdm(list(hlt_dct)):
        for pn in list(hlt_dct[pdf]):
            for sn in list(hlt_dct[pdf][pn]):
                sent = text_cleaning(hlt_dct[pdf][pn][sn]["sentence"])
                preprocessed_text = preprocess_spanish_text(sent)
                sents = get_nltk_sents(preprocessed_text, tokenizer, abbrevs)
                postprocessed_sents = remove_short_sents(sents, min_num_words)
                hlt_dct[pdf][pn][sn]["sentence"] = " ".join(postprocessed_sents)
    return hlt_dct

def paragraph_to_sents(hlt_dct):
    '''
    cleans highlights, splits them into sentences, then attaches the same label to each subsentence
    '''
    abbrevs= None
    tokenizer = ES_TOKENIZER
    min_num_words = 5
    for pdf in tqdm(list(hlt_dct)):
        for pn in list(hlt_dct[pdf]):
            new_page = {}
            for sn in list(hlt_dct[pdf][pn]):
                sent = text_cleaning(hlt_dct[pdf][pn][sn]["sentence"])
                preprocessed_text = preprocess_spanish_text(sent)
                sents = get_nltk_sents(preprocessed_text, tokenizer, abbrevs)
                postprocessed_sents = remove_short_sents(sents, min_num_words)
                for i in range(len(postprocessed_sents)):
                    new_page[i] = {
                        "sentence": postprocessed_sents[i],
                        "label": hlt_dct[pdf][pn][sn]["label"]
                    }
            hlt_dct[pdf][pn]=new_page
    return hlt_dct

def remove_empty_hlts(empties):
    for pdf in list(empties):
        for pg in list(empties[pdf]):
            for snt in list(empties[pdf][pg]):
                if len(empties[pdf][pg][snt]["sentence"]) == 0:
                    empties[pdf][pg].pop(snt)
            if len(empties[pdf][pg].keys()) == 0:
                empties[pdf].pop(pg)
        if len(empties[pdf].keys()) == 0:
            empties.pop(pdf)
    return empties

###################################################
# Now convert current dictionary structure into sentence-label structure
###################################################

def hlt_parse(hlts):
    sentences=[]
    labels=[]
    for pdf in list(hlts):
        for pn in list(hlts[pdf]):
            for sn in list(hlts[pdf][pn]):
                sentences.append(hlts[pdf][pn][sn]["sentence"])
                labels.append(hlts[pdf][pn][sn]["label"])
    return sentences, labels

###################################################
# get sentence, label lists from pdf annots
###################################################
def get_annot_sentlabs(pdf_annots):
    pdf_annots = clean_labels(pdf_annots)
    pdf_annots= remove_empty_labels(pdf_annots)
    pdf_annots = sentcheck_dups(pdf_annots)
    pdf_annots = keep_paragraph(pdf_annots)
    #pdf_annots = paragraph_to_sents(pdf_annots)
    pdf_annots = remove_empty_hlts(pdf_annots)
    sentences, labels = hlt_parse(pdf_annots)
    print(f'{len(sentences)} sent / {len(labels)} labels')
    return sentences, labels

if __name__ == '__main__':
    #input_path= "C:/Users/Allie/Documents/GitHub/policy-classifier/populate_corpora/outputs/pdf_texts.json"
    basedir = os.getcwd()
    output_path = basedir+"\\outputs"
    input_path= basedir+"\\outputs\\IrishPoliciesMar24.json"
    #ES_TOKENIZER = nltk.data.load("tokenizers/punkt/spanish.pickle")
    ES_TOKENIZER = []
    EN_TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")
    with open(input_path,"r", encoding="utf-8") as f:
        pdf_texts = json.load(f)
    
    #######################################
    # To get clean text of pdf texts:
    ##################################
    #sents = get_clean_text_sents(pdf_texts)
    texts = format_fulltxt_for_json(pdf_texts)

    with open(os.path.join(output_path, 'IrishPolsToLabel.json'), 'w', encoding="utf-8") as outfile:
        json.dump(texts, outfile, ensure_ascii=False, indent=4)
    ''''''
    #######################################################
    # To get clean text of pdf annots:
    ##################################
    '''
    sentences, labels = get_annot_sentlabs(pdf_texts)
    
    with open(os.path.join(output_path, 'annot_sents.json'), 'w', encoding="utf-8") as outfile:
        json.dump(sentences, outfile, ensure_ascii=False, indent=4)
    with open(os.path.join(output_path, 'annot_labels.json'), 'w', encoding="utf-8") as outfile:
        json.dump(labels, outfile, ensure_ascii=False, indent=4)
    '''
    ## need to fix the fact more than one label translates to the next highlight being 
   