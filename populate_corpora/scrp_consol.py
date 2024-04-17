import json
import os
from pathlib import Path
from tqdm import tqdm
#
from typing import Dict, List, Any, Set
import nltk
#nltk.download('punkt')
import unidecode
from utils import *
#
from io import BytesIO
from zipfile import ZipFile
from PyPDF2 import PdfReader
import glob
import time

from json_cleaning import preprocess_english_text

##########################################################################################
#   Get full text of PDFs
##########################################################################################

def scrp_itm_to_fulltxt(itm_col):
    '''
    takes a collection of scrapy items yeilded from spider
    '''
    big_dct = {}
    for dct in itm_col:
        big_dct[dct["hash_name"]] = {
            "doc_title":dct["doc_title"],
            "pg_title":dct["pg_title"],
            "pg_link":dct["pg_link"],
            "publication_date":dct["publication_date"],
            "department":dct["department"],
            "type":dct["type"],
            "doc_link":dct["file_urls"][0]
        }
    error_hash = []
    too_big = []
    for hash in tqdm(big_dct):
        try:
            file = os.path.join(INPUT_DIR, hash+'.pdf')
            if os.path.getsize(file) > 65000000:
                too_big.append(hash)
                raise RuntimeError(f"PyPDF2 cannot handle this large of a file: {hash}")
            pdfReader = PdfReader(file)  # read file
            # doc_dict holds the attributes of each pdf file
            big_dct[hash]["text"]=""
            for page in range(len(pdfReader.pages)):
                page_text = pdfReader.pages[page].extract_text()  # extracting pdf text
                #page_text = text_cleaning(page_text)  # clean pdf text
                big_dct[hash]["text"] += page_text  # concatenate pages' text
            big_dct[hash]["text"] = preprocess_english_text(big_dct[hash]["text"])
        except Exception as e:  # In case the file is corrupted
            print(f"Could not read {hash} due to {e}")
            error_hash.append(hash)
    for hash in error_hash:
        big_dct.pop(hash)
        print(f"Removed hash {hash} from dictionary")
    for hash in too_big:
        print(f"{hash} could not be processed because it was too big")
    return big_dct

def main():
    start = time.time()
    with open(INPUT_PTH,"r", encoding="utf-8") as f:
        itm_json = json.load(f)

    pdf_dict = scrp_itm_to_fulltxt(itm_json)

    #sent_dct = get_clean_text_dct(pdf_dict, EN_TOKENIZER)

    with open(OUTPUT_PTH+"/scraped_pdfs.json", 'w', encoding="utf-8") as outfile:
        json.dump(pdf_dict, outfile, ensure_ascii=False, indent=4)
    print(f"Total time: {time.time()-start}")

if __name__ == '__main__':
    OUTPUT_PTH = "C:/Users/Allie/Documents/GitHub/policy-classifier/policy_scraping/outputs"
    INPUT_PTH= "C:/Users/Allie/Documents/GitHub/policy-classifier/policy_scraping/outputs/govie.json"
    INPUT_DIR= "C:/Users/Allie/Documents/GitHub/policy-classifier/policy_scraping/outputs/full"

    EN_TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")
    '''
    with open(input_path,"r", encoding="utf-8") as f:
        pdf_texts = json.load(f)
    '''
    main()

    '''
    with open(os.path.join(output_path, 'annot_sents.json'), 'w', encoding="utf-8") as outfile:
        json.dump(sentences, outfile, ensure_ascii=False, indent=4)
    with open(os.path.join(output_path, 'annot_labels.json'), 'w', encoding="utf-8") as outfile:
        json.dump(labels, outfile, ensure_ascii=False, indent=4)
    '''
   