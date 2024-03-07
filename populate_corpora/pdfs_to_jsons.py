# -*- coding: utf-8 -*-
"""
Based on original repo's /tasks/extract_text/src/make_pdfs.py which extracted text from onedrive_docs.zip (which contained pdf files separated into folders by country) and 
outputted a json file of the pdf names and their (partially cleaned) extracted text.

In this version, one function to get full text from folder of pdfs, one function to get full text from zip file of pdfs > outputs are dictionaries.
Leaving the file saving for another function so that it's easier to just pass the dictionaries between functions instead of file I/O 
Also, adding my own functions to extract annotation comments and highlights.
Leaving text cleaning for later.
"""
import json
import os
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import argparse
from PyPDF2 import PdfReader
from tqdm import tqdm
import glob

def text_cleaning(text):
    """
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

def txt_to_dct(pdfReader):
    '''
    Input: 
        pdfReader (PyPDF2 object): Reader in use in loop
    Returns:
        doc_dict (dct): dictionary of single pdf with text
    '''
    doc_dict = {i[1:]: str(j) for i, j in pdfReader.metadata.items()}
    #doc_dict["Country"] = file.split("_")[-1][:-4]
    doc_dict["Text"]=""
    for page in range(len(pdfReader.pages)):
        page_text = pdfReader.pages[page].extract_text()  # extracting pdf text
        page_text = text_cleaning(page_text)  # clean pdf text
        doc_dict["Text"] += page_text  # concatenate pages' text
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
            try:
                pdfReader = PdfReader(file)  # read file
                # doc_dict holds the attributes of each pdf file
                doc_dict = txt_to_dct(pdfReader)
                pdf_dict[os.path.splitext(os.path.basename(file))[0]] = doc_dict
            except Exception as e:  # In case the file is corrupted
                errors.append(f"Could not read {file} due to {e}")
    print(errors)
    print(f"Successfully extracted {len(pdf_dict)}/{len(filenames)} pdfs")
    return pdf_dict

if __name__ == '__main__':
    input_zip = "C:/Users/Allie/Documents/GitHub/policy-classifier/populate_corpora/pdf_input/onedrive_docs.zip"
    output_path = "C:/Users/Allie/Documents/GitHub/policy-classifier/populate_corpora/outputs"
    input_dir= "C:/Users/Allie/Documents/GitHub/policy-classifier/populate_corpora/pdf_input/latam_pols"

    pdf_dict = pdfs_to_txt_dct(input_dir)
    with open(os.path.join(output_path, 'pdf_texts.json'), 'w', encoding="utf-8") as outfile:
        json.dump(pdf_dict, outfile, ensure_ascii=False, indent=4)


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
