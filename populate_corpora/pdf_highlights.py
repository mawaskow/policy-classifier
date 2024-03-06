# https://medium.com/@vinitvaibhav9/extracting-pdf-highlights-using-python-9512af43a6d
# there is a bit of noise, other text getting scraped in from the highlight coordinates, and duplications of text.

import PyPDF2
import os
from os import listdir, walk
from os.path import isfile, join
from zipfile import ZipFile
from collections import defaultdict
from io import BytesIO
import glob
import fitz

experiment = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\extract_text\\input\\onedrive_docs\\Chile\\Decreto 51_Chile.pdf"

doc = fitz.open(experiment)
# Total page in the pdf
#print(len(doc))
# taking page for further processing
page = doc[1]
'''
for page in doc:
    print(page)
    for annot in page.annots():
        print(annot.get_text("text"))
        print(annot.get_textbox(annot.rect))
'''

# list to store the co-ordinates of all highlights
highlights = []
# loop till we have highlight annotation in the page
annot = page.first_annot
#
while annot:
    if annot.type[0] == 8:
        all_coordinates = annot.vertices
        if len(all_coordinates) == 4:
            highlight_coord = fitz.Quad(all_coordinates).rect
            highlights.append(highlight_coord)
        else:
            all_coordinates = [all_coordinates[x:x+4] for x in range(0, len(all_coordinates), 4)]
            for i in range(0,len(all_coordinates)):
                coord = fitz.Quad(all_coordinates[i]).rect
                highlights.append(coord)
    annot = annot.next

all_words = page.get_text("words")

# List to store all the highlighted texts
highlight_text = []
for h in highlights:
    sentence = [w[4] for w in all_words if fitz.Rect(w[0:4]).intersects(h)]
    highlight_text.append(" ".join(sentence))

print(" ".join(highlight_text))