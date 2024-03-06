import PyPDF2
import os
from os import listdir, walk
from os.path import isfile, join
from zipfile import ZipFile
from collections import defaultdict
from io import BytesIO
import glob

def pdf_comments_to_dct(file_path):
    """
    This function extracts comments from a PDF file and returns them as a list.
    Parameters:
    file_path (str): The path to the PDF file
    Returns:
    dct: A dct of comments extracted from the PDF file
    """
    com_dct = {}
    try:
        # Open the PDF file
        with open(file_path, 'rb') as pdf_file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            # Get the number of pages in the PDF file
            num_pages = len(pdf_reader.pages)
            # Initialize an empty list to store comments
            #comments = []
            #coords = []
            # Loop through each page in the PDF file
            for page_num in range(num_pages):
                # Get the page object
                page = pdf_reader.pages[page_num]
                # Get the annotations for the page
                if "/Annots" in page:
                    for annot in page["/Annots"]:
                        subtype = annot.get_object()["/Subtype"]
                        if subtype == "/Highlight":
                            i = 0
                            com_dct[page_num] = {}
                            com_dct[page_num][i] = {}
                            com_dct[page_num][i]["type"] = subtype
                            cos = annot.get_object()["/QuadPoints"]
                            #x1, y1, x2, y2, x3, y3, x4, y4 = coords
                            com_dct[page_num][i]['coords'] = cos
                            try:
                                comment = annot.get_object()["/Contents"]
                                com_dct[page_num][i]['text'] = comment
                            except:
                                pass
                            i+=1
            # Return the dct of comments
            return com_dct
    except Exception as e:
        # Log the error
        print(f"Error: {e}")
        return []
    
def pdf_comments_to_lst(file_path):
    """
    This function extracts comments from a PDF file and returns them as a list.
    Parameters:
    file_path (str): The path to the PDF file
    Returns:
    list: A list of comments extracted from the PDF file
    """
    try:
        # Open the PDF file
        with open(file_path, 'rb') as pdf_file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            # Get the number of pages in the PDF file
            num_pages = len(pdf_reader.pages)
            # Initialize an empty list to store comments
            comments = []
            # Loop through each page in the PDF file
            for page_num in range(num_pages):
                # Get the page object
                page = pdf_reader.pages[page_num]
                # Get the annotations for the page
                if "/Annots" in page:
                    for annot in page["/Annots"]:
                        try:
                            comment = annot.get_object()["/Contents"]
                            comments.append(comment)
                        except:
                            pass
            # Return the dct of comments
            return comments
    except Exception as e:
        # Log the error
        print(f"Error: {e}")
        return []


INPUT_PATH = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\extract_text\\input\\onedrive_docs\\"
filenames = []
dir_path = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\extract_text\\input\\onedrive_docs\\**\\*.*"
for file in glob.glob(dir_path, recursive=True):
    filenames.append(file)
pdf_dict = defaultdict(dict)
for file in filenames:
    print(f"Processing {file}...")
    try:
        cmts = pdf_comments_to_dct(os.path.join(INPUT_PATH, file))
    except Exception as e:  # In case the file is corrupted
        print(e)
        print(f"Attempting to recover {file}...")
        #pdfReader = file_recovery(file, myzip)  # attempting to recover file


experiment = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\extract_text\\input\\onedrive_docs\\Chile\\2019CVE 1713470_Chile.pdf"

cmts = []

pdf_dict = defaultdict(dict)

print(f"Processing {experiment}...")
try:
    cmts = pdf_comments_to_lst(experiment)
    #cmts = pdf_comments_to_dct(experiment)
except Exception as e:  # In case the file is corrupted
    print(e)
    print(f"Attempting to recover experiment...")
    #pdfReader = file_recovery(file, myzip)  # attempting to recover file
print(cmts)
