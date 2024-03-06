# -*- coding: utf-8 -*-
"""
Extracts text and metadata from pdf files. Reads pdf files from ../input/onedrive_docs.zip 
and outputs extracted information to ../output/pdf_files.json.
"""
import json
import logging
import os
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import argparse
from pikepdf import Pdf
from PyPDF2 import PdfReader


def text_cleaning(text):
    """Cleans a piece of text by removing escaped characters.

    Args:
        text (str): string with text

    Returns:
        str: cleaned piece of text
    """
    # Remove escaped characters
    escapes = ''.join([chr(char) for char in range(1, 32)])
    text = text.translate(str.maketrans('', '', escapes))
    return text


def file_recovery(filename, zipfile, output_path):
    """Attempts to recover a pdf file inside a zip that wasn't read correctly.

    Args:
        filename (str): file path string
        zipfile (ZipFile): ZipFile instance where file is located

    Returns:
        PdfReader: new PdfReader instance from where text will be read
    """
    with Pdf.open(BytesIO(zipfile.read(filename))) as pdf_file:  # attempting to recovery file
        path_to_file = os.path.join(output_path, os.path.basename(filename))
        pdf_file.save(path_to_file)  # writes the file to disk
        pdfReader = PdfReader(path_to_file)  # attempts to read the file again
        os.remove(path_to_file)
    return pdfReader

def text_reader():
    """Function to solve TypeError: ord() expected string of length 1, but 
     int found error. TO IMPLEMENT.
    """
    # Useful link: https://stackoverflow.com/questions/55993860/getting-typeerror-ord-expected-string-of-length-1-but-int-found-error
    pass

def main(input_zip, output_path):
    logger = logging.getLogger(__name__)
    logger.info('Making pdf_files.json from base pdf files')

    with ZipFile(input_zip) as myzip:
        # List files inside zip
        filenames = list(map(lambda x: x.filename, filter(lambda x: not x.is_dir(), myzip.infolist())))
        pdf_dict = defaultdict(dict)
        for file in filenames:
            logger.info(f"Processing {file}...")
            try:
                pdfReader = PdfReader(BytesIO(myzip.read(file)))  # read file
                # doc_dict holds the attributes of each pdf file
                doc_dict = {i[1:]: str(j) for i, j in pdfReader.metadata.items()}
                doc_dict["Country"], doc_dict["Text"] = file.split("/")[0], ""
                for page in range(len(pdfReader.pages)):
                    try:
                        page_text = pdfReader.pages[page].extract_text()  # extracting pdf text
                    except TypeError as e:
                        logger.warning(e)
                        logger.info(f"Skipping {file}...")
                        continue
                    page_text = text_cleaning(page_text)  # clean pdf text
                    doc_dict["Text"] += page_text  # concatenate pages' text
                pdf_dict[os.path.splitext(os.path.basename(file))[0]] = doc_dict
            except Exception as e:  # In case the file is corrupted
                logger.warning(e)
                logger.info(f"Attempting to recover {file}...")
                pdfReader = file_recovery(file, myzip, output_path)  # attempting to recover file

    with open(os.path.join(output_path, 'pdf_files.json'), 'w', encoding="utf-8") as outfile:
        try:
            json.dump(pdf_dict, outfile, ensure_ascii=False, indent=4)
        except UnicodeEncodeError as e:
            logger.warning(e)
            logger.info(f"Skipping...")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    '''
    input_zip = "C:/Users/Ales/Documents/GitHub/policy-data-analyzer/tasks/extract_text/input/onedrive_docs.zip"
    output_path = "C:/Users/Ales/Documents/GitHub/policy-data-analyzer/tasks/extract_text/output"
    main(input_zip, output_path)
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
    
