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

###################################################
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

if __name__ == '__main__':
    output_path = "C:/Users/Allie/Documents/GitHub/policy-classifier/populate_corpora/outputs"
    input_path= "C:/Users/Allie/Documents/GitHub/policy-classifier/populate_corpora/outputs/pdf_texts.json"

    pdf_dict = []
    ES_TOKENIZER = nltk.data.load("tokenizers/punkt/spanish.pickle")

    with open(input_path,"r", encoding="utf-8") as f:
        pdf_texts = json.load(f)
    
    sentences = get_clean_text_sents(pdf_texts)

    with open(os.path.join(output_path, 'full_text_sents.json'), 'w', encoding="utf-8") as outfile:
        json.dump(sentences, outfile, ensure_ascii=False, indent=4)