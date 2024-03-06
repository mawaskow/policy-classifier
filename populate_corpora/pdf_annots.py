import PyPDF2
import os
import glob
import fitz
import json
import time
import argparse
    
def text_cleaning(text):
    """Cleans a piece of text by removing escaped characters. (from src make_pfs)

    Args:
        text (str): string with text

    Returns:
        str: cleaned piece of text
    """
    # Remove escaped characters
    escapes = ''.join([chr(char) for char in range(1, 32)])
    text = text.translate(str.maketrans('', '', escapes))
    return text

def pdf_comments_to_sim_dct(file_path):
    """
    This function extracts comments from a PDF file and returns them as a list.
    Parameters:
    file_path (str): The path to the PDF file
    Returns:
    list: A list of comments extracted from the PDF file
    """
    simple_dct = {}
    try:
        # Open the PDF file
        with open(file_path, 'rb') as pdf_file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
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
                                simple_dct[page_num] = {}
                            simple_dct[page_num][i] = comment
                            i+=1
                        except:
                            pass
            # Return the dct of comments
            return simple_dct
    except Exception as e:
        # Log the error
        print(f"Error: {e}")
        return {}

def pdf_highlight_to_dct(file_path):
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
                highlt_dct[page_num][i] = text_cleaning(" ".join(highlight_text))
            i+=1
            # this throws No attribute .parent: type(self)=<class 'fitz.Annot'> id(self)=#########: have set id(self.parent)=#######.
            annot = annot.next
    return highlt_dct

def main(input_path, output_path):
    dir_path = input_path+"\\**\\*.*"
    filenames = []
    pdf_dct = {}
    for file in glob.glob(dir_path, recursive=True):
        filenames.append(file)
    start = time.time()
    #for each file
    for file in filenames:
        fname = file.split('\\')[-1][:-4]
        print(f"Processing {fname}...")
        pdf_dct[fname] = {}
        # get comment annotation and highlighted text dictionaries
        # first key is page number, then the number of each sentence/annotation
        try:
            cmts = pdf_comments_to_sim_dct(os.path.join(input_path, file))
            hlts = pdf_highlight_to_dct(os.path.join(input_path, file))
            '''
            cfile = os.path.join(output_path, fname+"_comm.json")
            hfile = os.path.join(output_path, fname+"_hlt.json")
            print(f"Writing to {cfile}")
            with open(cfile, 'w+') as fp:
                json.dump(cmts, fp, indent=4)
            print(f"Writing to {hfile}")
            with open(hfile, 'w+') as fp:
                json.dump(hlts, fp, indent=4)
            '''
        except Exception as e:  # In case the file is corrupted
            print(e)
            print(f"Attempting to recover {file}...")
            #pdfReader = file_recovery(file, myzip)  # attempting to recover file
        for p in cmts.keys():
            if p in hlts.keys():
                pdf_dct[fname][p]={}
                for i in cmts[p].keys():
                    if i in hlts[p].keys():
                        pdf_dct[fname][p][i] = {}
                        pdf_dct[fname][p][i]["sentence"]= hlts[p][i]
                        # label cleaning
                        label = cmts[p][i].split("\n")[0]
                        label = label.split("\r")[0][3:]
                        pdf_dct[fname][p][i]["label"] = label
                    else:
                        print(fname, "did not have same highlight count for page:", p)
            else:
                print(fname, "did not have highlight for page:", p)
    name = "pdf_extract.json"
    file = os.path.join(output_path, name)
    print(f"Writing to {file}")
    with open(file, 'w+') as fp:
        json.dump(pdf_dct, fp, indent=4)

    print(f"All done in {round(time.time()-start)}s")

if __name__ == '__main__':
    
    input_path = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\extract_text\\input\\onedrive_docs"
    output_path = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\extract_text\\output"
    main(input_path, output_path)

    # cmd arg
    # python ./src/pdf_annots.py -i "C:\\Users\\ales\\Documents\\GitHub\\policy-data-analyzer\\tasks\\extract_text\\input\\onedrive_docs" -o "C:\\Users\\ales\\Documents\\GitHub\\policy-data-analyzer\\tasks\\extract_text\\output"
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_folder', required=True,
                        help="Path to input folder of pdfs (NOT zipped)")
    parser.add_argument('-o', '--output_folder', required=True,
                        help="Path to where output should be stored.")

    args = parser.parse_args()

    main(args.input_folder, args.output_folder)
    '''