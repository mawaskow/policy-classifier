# -*- coding: utf-8 -*-
"""
Based on original repo's /tasks/extract_text/src/make_pdfs.py which extracted text from onedrive_docs.zip (which contained pdf files separated into folders by country) and 
outputted a json file of the pdf names and their (partially cleaned) extracted text.

In this version: one function to get full text from folder of pdfs, one function to get full text from zip file of pdfs > outputs are dictionaries.
Leaving the file saving for another function so that it's easier to just pass the dictionaries between functions instead of file I/O 
Also, adding my own functions to extract annotation comments and highlights.
Leaving text cleaning for later.

Eventually adapt PDF annotation fxns to handle zip file inputs
"""
import json
import os
from io import BytesIO
from zipfile import ZipFile
from PyPDF2 import PdfReader
from tqdm import tqdm
import glob
import nltk

##########################################################################################
#   Get full text of PDFs
##########################################################################################

def txt_to_dct(pdfReader):
    '''
    Input: 
        pdfReader (PyPDF2 object): Reader in use in loop
    Returns:
        doc_dict (dct): dictionary of single pdf with text
    '''
    doc_dict = {}
    doc_dict["text"]=""
    for page in range(len(pdfReader.pages)):
        page_text = pdfReader.pages[page].extract_text()  # extracting pdf text
        #page_text = text_cleaning(page_text)  # clean pdf text
        doc_dict["text"] += page_text  # concatenate pages' text
    return doc_dict

def pdfs_to_txt_dct(file_dir):
    '''
    Input:
        input_path (str): path directory or zip folder of pdfs
    Output:
        error messages
    Returns:
        pdf_dict (dct): dictionary of pdfs text
    '''
    errors = []
    filenames = []
    pdf_dict = {}
    if file_dir[-4:]== ".zip":
        with ZipFile(file_dir) as myzip:
            filenames = list(map(lambda x: x.filename, filter(lambda x: not x.is_dir(), myzip.infolist())))
            for file in tqdm(filenames):
                key = os.path.splitext(os.path.basename(file))[0]
                try:
                    pdfReader = PdfReader(BytesIO(myzip.read(file)))
                    # doc_dict holds the attributes of each pdf file
                    doc_dict = txt_to_dct(pdfReader)
                    pdf_dict[os.path.splitext(os.path.basename(file))[0]] = doc_dict
                except Exception as e:  # In case the file is corrupted
                    errors.append(f"Could not read {file} due to {e}")
    else:
        file_dir = file_dir+"\\**\\*.*"
        for file in glob.glob(file_dir, recursive=True):
            filenames.append(file)
        for file in tqdm(filenames):
            key = os.path.splitext(os.path.basename(file))[0]
            try:
                pdfReader = PdfReader(file)  # read file
                # doc_dict holds the attributes of each pdf file
                doc_dict = txt_to_dct(pdfReader)
                pdf_dict[key] = doc_dict
            except Exception as e:  # In case the file is corrupted
                errors.append(f"Could not read {file} due to {e}")
    print(errors)
    print(f"Successfully extracted {len(pdf_dict)}/{len(filenames)} pdfs")
    return pdf_dict

def scrp_itm_to_fulltxt(meta_dct, filedir):
    '''
    takes metadata info/ scrapy items, and directory of downloaded files
    '''
    big_dct = {}
    for dct in meta_dct:
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
            file = os.path.join(filedir, hash+'.pdf')
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
    basedir = os.getcwd()
    output_path = basedir+"\\populate_corpora\\outputs"
    filedir= basedir+"\\policy_scraping\\policy_scraping\\outputs\\forestry\\full"
    pdf_info_addr = basedir+"\\policy_scraping\\outputs\\goviefor.json"
    
    with open(pdf_info_addr, "r", encoding="utf-8") as f:
        meta_dct = json.load(f)
    
    pdf_dict = scrp_itm_to_fulltxt(meta_dct, filedir)
    
    with open(os.path.join(output_path, 'ForestryPolicies.json'), 'w', encoding="utf-8") as outfile:
        json.dump(pdf_dict, outfile, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    EN_TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")
    main()
    '''
    # python ./src/make_pdfs.py -i "C:/Users/Ales/Documents/GitHub/policy-data-analyzer/tasks/extract_text/input/onedrive_docs.zip" -o "C:/Users/Ales/Documents/GitHub/policy-data-analyzer/tasks/extract_text/output"
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_zip', required=True,
                        help="Path to zip folder including input files.")
    parser.add_argument('-o', '--output_path', required=True,
                        help="Path to folder where output will be saved.")

    args = parser.parse_args()

    main(args.input_zip, args.output_path)
    '''
