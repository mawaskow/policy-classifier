# -*- coding: utf-8 -*-
"""
The first part of this code is based on original repo's /tasks/text_preprocessing/src/sentence_split_local.py 
which would clean and split the pdf texts into sentences as well
as format them for processing outputted a json file of the pdf names and their (partially cleaned) extracted text.
>sents_json = {file_id: {"metadata":
>                {"n_sentences": len(postprocessed_sents),
>                "language": language},
>                "sentences": postprocessed_sents}}
>with open(os.path.join(output_dir, f'{file_id}_sents.json'), 'w') as f:

The second part of this code is new, cleaning and preparing data from doccano jsons
"""
import json
import os
from tqdm import tqdm
from typing import List, Set
import nltk
#nltk.download('punkt')
import unidecode
import re
from rapidfuzz import fuzz

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

###################################
#      UTILS from prev repo utils.py file
###################################

def remove_html_tags(text: str) -> str:
    """Remove html tags from a string"""
    return re.sub(re.compile('<.*?>'), '', text)

def replace_links(text: str) -> str:
    text = re.sub(r'http\S+', '[URL]', text)
    return re.sub(r'www\S+', '[URL]', text)

def remove_multiple_spaces(text: str) -> str:
    return re.sub('\s+', ' ', text)

def parse_emails(text: str) -> str:
    """
    Remove the periods from emails in text, except the last one
    """
    emails = [email if email[-1] != "." else email[:-1] for email in re.findall(r"\S*@\S*\s?", text)]
    for email in emails:
        new_email = email.replace(".", "")
        text = text.replace(email, new_email)
    return text

def parse_acronyms(text: str) -> str:
    """
    Remove the periods from acronyms in the text (i.e "U.S." becomes "US")
    """
    acronyms = re.findall(r"\b(?:[a-zA-Z]\.){2,}", text)
    for acronym in acronyms:
        new_acronym = acronym.replace(".", "")
        text = text.replace(acronym, new_acronym)
    return text

##########################################################################################
#   preprocessing from orig pipeline sentence_split_local,py file
##########################################################################################

def preprocess_text(txt: str, remove_new_lines: bool = False) -> str:
    """
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
    return preprocess_text(txt, remove_new_lines)

def preprocess_spanish_text(txt: str, remove_new_lines: bool = False) -> str:
    return unidecode.unidecode(preprocess_text(txt, remove_new_lines))

def remove_short_sents(sents: List[str], min_num_words: int = 4) -> List[str]:
    return [sent for sent in sents if len(sent.split()) >= min_num_words]

def get_nltk_sents(txt: str, tokenizer: nltk.PunktSentenceTokenizer, extra_abbreviations: Set[str] = None) -> List[str]:
    if extra_abbreviations is not None:
        tokenizer._params.abbrev_types.update(extra_abbreviations)
    return tokenizer.tokenize(txt)

def format_sents_for_output(sents, doc_id):
    formatted_sents = {}
    for i, sent in enumerate(sents):
        formatted_sents.update({f"{doc_id}_sent_{i}": {"text": sent, "label": []}})
    return formatted_sents

#####################################
# New addtns
####################################

def format_sents_for_doccano(sents):
    formatted_sents = []
    for sent in sents:
        formatted_sents.append({"text": sent, "label": []})
    return formatted_sents

def get_clean_text_sents(pdf_conv, tokenizer):
    '''
    Takes a dictionary of full text of pdf files and returns all sentences, cleaned, in one list
    Input:
        pdf_conv (dct): dictionary of full text of pdf files
    Output: 
        Error files
    Returns:
        sentences (lst): all sentences, cleaned
    '''
    abbrevs= None
    min_num_words = 5
    sentences = []
    file_lst = []
    for key in pdf_conv:
        file_lst.append((key,pdf_conv[key]['text']))
    error_files={}
    i = 0
    for file_id, text in tqdm(file_lst):
        try:
            preprocessed_text = preprocess_english_text(text)
            sents = get_nltk_sents(preprocessed_text, tokenizer, abbrevs)
            postprocessed_sents = remove_short_sents(sents, min_num_words)
            sentences+=postprocessed_sents
        except Exception as e:
            error_files[str(file_id)]= str(e)
        i += 1
    print(f'Number of error files: {len(error_files)}')
    return sentences

def prelabeling(dcc_lst):
    kwds = ["forest", "incentive", "instrument", "tree", "scheme", "grant", "pay", "loan", "credit", "subsid", 
            "cash", "restor", "tax", "train", "assist", "support", "penal", "compensat", "expert", "fine"]
    flt_lst =[]
    for entry in dcc_lst:
        if any(kwd in entry['text'].lower() for kwd in kwds):
            flt_ntr = {}
            flt_ntr['text'] = entry['text']
            flt_ntr["label"] = ["non-incentive"]
            if any(kwd in entry['text'].lower() for kwd in ["grant", "pay", "cash"]):
                flt_ntr['label'] = ["mention-direct payment"]
            if any(kwd in entry['text'].lower() for kwd in ["loan","lend", "credit", "insur", "guarantee","debt"]):
                flt_ntr['label'] = ["mention-credit"]
            if any(kwd in entry['text'].lower() for kwd in ["tax", "liabilit", "deduct"]):
                flt_ntr['label'] = ["mention-tax deduction"]
            if any(kwd in entry['text'].lower() for kwd in ["train", "assist","expert"]):
                flt_ntr['label'] = ["mention-technical assistance"]
            if any(kwd in entry['text'].lower() for kwd in ["supplies", "equip", "infrastructure"]):
                flt_ntr['label'] = ["mention-supplies"]
            if any(kwd in entry['text'].lower() for kwd in ["penal", "fine"]):
                flt_ntr['label'] = ["mention-fine"]
            flt_lst.append(flt_ntr)
    return flt_lst

def dcno_to_only_sents(dcno_json):
    return [entry['text'] for entry in dcno_json]

####################
# new doccano json output file cleaning
####################

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
        #for i in range(2):
        #    n = random.randint(0, len(sents))
        #    print(f'[{n}] {labels[n]}: {sents[n]}')
    return sents, labels

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
        #n = random.randint(0, len(inc))
        #print(f'[{n}] Incentive: {inc[n]}')
        #n = random.randint(0, len(noninc))
        #print(f'[{n}] Non-Incentive: {noninc[n]}')
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
        #for i in range(5):
        #    n = random.randint(0, len(mc_sents))
        #    print(f'[{n}] {mc_labels[n]}: {mc_sents[n]}')
    return mc_sents, mc_labels

def main():
    basedir = os.getcwd()
    output_path = basedir+"\\populate_corpora\\outputs\\"
    input_path= basedir+"\\populate_corpora\\outputs\\ForestryPolicies.json"
    with open(input_path,"r", encoding="utf-8") as f:
        pdf_texts = json.load(f)
    output= get_clean_text_sents(pdf_texts, EN_TOKENIZER)
    output = format_sents_for_doccano(output)
    output = prelabeling(output)
    with open(os.path.join(output_path, 'Forestry_prelab.json'), 'w', encoding="utf-8") as outfile:
        json.dump(output, outfile, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    EN_TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")
    main()
   