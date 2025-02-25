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
import argparse
from PyPDF2 import PdfReader
import fitz
from tqdm import tqdm
import glob

##########################################################################################
#   Get full text of PDFs
##########################################################################################

def txt_to_dct_meta(pdfReader, pdf_meta):
    '''
    Input: 
        pdfReader (PyPDF2 object): Reader in use in loop
    Returns:
        doc_dict (dct): dictionary of single pdf with text
    '''
    doc_dict = pdf_meta
    doc_dict["text"]=""
    for page in range(len(pdfReader.pages)):
        page_text = pdfReader.pages[page].extract_text()  # extracting pdf text
        #page_text = text_cleaning(page_text)  # clean pdf text
        doc_dict["text"] += page_text  # concatenate pages' text
    return doc_dict

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

def pdfs_to_txt_dct(input_path):
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
    if input_path[-4:]== ".zip":
        with ZipFile(input_path) as myzip:
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
        input_path = input_path+"\\**\\*.*"
        for file in glob.glob(input_path, recursive=True):
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

def pdfs_to_txt_meta_dct(input_path, meta_dct):
    '''
    Input:
        input_path (str): path directory or zip folder of pdfs
        meta_dct
    Output:
        error messages
    Returns:
        pdf_dict (dct): dictionary of pdfs text
    '''
    errors = []
    filenames = []
    pdf_dict = {}
    if input_path[-4:]== ".zip":
        with ZipFile(input_path) as myzip:
            filenames = list(map(lambda x: x.filename, filter(lambda x: not x.is_dir(), myzip.infolist())))
            for file in tqdm(filenames):
                key = os.path.splitext(os.path.basename(file))[0]
                try:
                    pdfReader = PdfReader(BytesIO(myzip.read(file)))
                    # doc_dict holds the attributes of each pdf file
                    doc_dict = txt_to_dct(pdfReader, meta_dct[key])
                    pdf_dict[os.path.splitext(os.path.basename(file))[0]] = doc_dict
                except Exception as e:  # In case the file is corrupted
                    errors.append(f"Could not read {file} due to {e}")
    else:
        input_path = input_path+"\\**\\*.*"
        for file in glob.glob(input_path, recursive=True):
            filenames.append(file)
        for file in tqdm(filenames):
            key = os.path.splitext(os.path.basename(file))[0]
            try:
                pdfReader = PdfReader(file)  # read file
                # doc_dict holds the attributes of each pdf file
                doc_dict = txt_to_dct(pdfReader, meta_dct[key])
                pdf_dict[key] = doc_dict
            except Exception as e:  # In case the file is corrupted
                errors.append(f"Could not read {file} due to {e}")
    print(errors)
    print(f"Successfully extracted {len(pdf_dict)}/{len(filenames)} pdfs")
    return pdf_dict

def restructure_data_from_scraper(json_dct):
    dct = {}
    for item in json_dct:
        dct[item['hash_name']] = {}
        dct[item['hash_name']]['pg_title'] = item['pg_title']
        dct[item['hash_name']]['pg_link'] = item['pg_link']
        dct[item['hash_name']]['publication_date'] = item['publication_date']
        dct[item['hash_name']]['department'] = item['department']
        dct[item['hash_name']]['type'] = item['type']
        dct[item['hash_name']]['doc_title'] = item['doc_title']
        dct[item['hash_name']]['file_urls'] = item['file_urls']
        
    return dct

def main():
    basedir = os.getcwd()
    output_path = basedir+"\\populate_corpora\\outputs"
    input_dir= basedir+"\\policy_scraping\\policy_scraping\\outputs\\forestry\\full"
    pdf_info_addr = basedir+"\\policy_scraping\\outputs\\goviefor.json"
    
    with open(pdf_info_addr, "r", encoding="utf-8") as f:
        pdf_info = json.load(f)
    
    pdf_info = restructure_data_from_scraper(pdf_info)

    pdf_dict = pdfs_to_txt_meta_dct(input_dir, pdf_info)
    
    with open(os.path.join(output_path, 'ForestryPolicies.json'), 'w', encoding="utf-8") as outfile:
        json.dump(pdf_dict, outfile, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()

    '''
    # cmd line example
    # python ./src/make_pdfs.py -i "C:/Users/Ales/Documents/GitHub/policy-data-analyzer/tasks/extract_text/input/onedrive_docs.zip" -o "C:/Users/Ales/Documents/GitHub/policy-data-analyzer/tasks/extract_text/output"
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_zip', required=True,
                        help="Path to zip folder including input files.")
    parser.add_argument('-o', '--output_path', required=True,
                        help="Path to folder where output will be saved.")

    args = parser.parse_args()

    main(args.input_zip, args.output_path)
    '''
