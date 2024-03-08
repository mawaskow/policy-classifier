# -*- coding: utf-8 -*-
"""
Based on original repo's /tasks/extract_text/src/make_pdfs.py which extracted text from onedrive_docs.zip (which contained pdf files separated into folders by country) and 
outputted a json file of the pdf names and their (partially cleaned) extracted text.

In this version: one function to get full text from folder of pdfs, one function to get full text from zip file of pdfs > outputs are dictionaries.
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
import fitz
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

def pdf_to_cmt_dct(file_path):
    """
    This function extracts comments from a PDF file and returns them as a dct.
    Parameters:
    file_path (str): The path to the PDF file
    Returns:
    list: A list of comments extracted from the PDF file
    """
    pdf_cmt_dct = {}
    try:
        # Open the PDF file
        with open(file_path, 'rb') as pdf_file:
            # Create a PDF reader object
            pdf_reader = PdfReader(pdf_file)
            # Get the number of pages in the PDF file
            num_pages = len(pdf_reader.pages)
            # Loop through each page in the PDF file
            for page_num in range(num_pages):
                # Get the page object
                page = pdf_reader.pages[page_num]
                i=0
                # Get the annotations for the page
                if "/Annots" in page:
                    for annot in page["/Annots"]:
                        try:
                            comment = annot.get_object()["/Contents"]
                            if i==0:
                                pdf_cmt_dct[page_num] = {}
                            pdf_cmt_dct[page_num][i] = comment
                            i+=1
                        except:
                            pass
    except Exception as e:
        print(f"Error: {e}")
    # Return the dct of comments
    return pdf_cmt_dct

def pdf_highlight_to_dct(file_path):
    # https://medium.com/@vinitvaibhav9/extracting-pdf-highlights-using-python-9512af43a6d
    # there is a bit of noise: other text getting scraped in from the highlight coordinates and duplications of text.
    # may want to look into other highlight/annotation extraction packages
    highlt_dct = {}
    doc = fitz.open(file_path)
    # traverse pdf by page
    for page_num in range(len(doc)):
        page = doc[page_num]
        # list of highlights for each page
        highlights = []
        annot = page.first_annot
        i=0
        while annot:
            if annot.type[0] == 8:
                all_coordinates = annot.vertices
                if len(all_coordinates) == 4:
                    highlight_coord = fitz.Quad(all_coordinates).rect
                    highlights.append(highlight_coord)
                else:
                    all_coordinates = [all_coordinates[x:x+4] for x in range(0, len(all_coordinates), 4)]
                    for j in range(0,len(all_coordinates)):
                        coord = fitz.Quad(all_coordinates[j]).rect
                        highlights.append(coord)
                    all_words = page.get_text("words")
            # List to store all the highlighted texts
            highlight_text = []
            for h in highlights:
                sentence = [w[4] for w in all_words if fitz.Rect(w[0:4]).intersects(h)]
                highlight_text.append(" ".join(sentence))
            if highlight_text:
                if i==0:
                    highlt_dct[page_num]={}
                #highlt_dct[page_num][i] = text_cleaning(" ".join(highlight_text))
                highlt_dct[page_num][i] = " ".join(highlight_text)
            i+=1
            annot = annot.next
    return highlt_dct

def pdfs_to_annot_dct(input_path):
    # can only handle normal folders right now
    dir_path = input_path+"\\**\\*.*"
    filenames = []
    pdf_dct = {}
    for file in glob.glob(dir_path, recursive=True):
        filenames.append(file)
    #
    #for each file
    for file in tqdm(filenames):
        fname = file.split('\\')[-1][:-4]
        print(f"Processing {fname}...")
        pdf_dct[fname] = {}
        # get comment annotation and highlighted text dictionaries
        # first key is page number, then the number of each sentence/annotation
        try:
            cmts = pdf_to_cmt_dct(os.path.join(input_path, file))
            hlts = pdf_highlight_to_dct(os.path.join(input_path, file))
            for p in cmts.keys():
                if p in hlts.keys():
                    pdf_dct[fname][p]={}
                    for i in cmts[p].keys():
                        if i in hlts[p].keys():
                            pdf_dct[fname][p][i] = {}
                            pdf_dct[fname][p][i]["sentence"]= hlts[p][i]
                            label = cmts[p][i]
                            pdf_dct[fname][p][i]["label"] = label
                        else:
                            print(f"{fname} did not have same highlight count for page {p}")
                else:
                    print(f"{fname} did not have highlight for page {p}")
        except Exception as e:
            print(f"{fname} was not processed due to: {e}")
    return pdf_dct

if __name__ == '__main__':
    input_zip = "C:/Users/Allie/Documents/GitHub/policy-classifier/populate_corpora/pdf_input/onedrive_docs.zip"
    output_path = "C:/Users/Allie/Documents/GitHub/policy-classifier/populate_corpora/outputs"
    input_dir= "C:/Users/Allie/Documents/GitHub/policy-classifier/populate_corpora/pdf_input/latam_pols"
    #pdf_dict = pdfs_to_txt_dct(input_dir)

    pdf_dict = pdfs_to_annot_dct(input_dir)
    with open(os.path.join(output_path, 'pdf_annots.json'), 'w', encoding="utf-8") as outfile:
        json.dump(pdf_dict, outfile, ensure_ascii=False, indent=4)


    '''
    # label cleaning
    label = cmts[p][i].split("\n")[0]
    label = label.split("\r")[0][3:]


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
