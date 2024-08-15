import pandas as pd
import hashlib
import json
import glob
import os

XLSX_FILE = "C:/Users/allie/Documents/PhD/IrishPoliciesMar24/pdfsinpols.xlsx"
OUTPUT_FILE = "C:/Users/allie/Documents/PhD/IrishPoliciesMar24/pdfsinpols.json"
INPUT_FOLDER = "C:/Users/allie/Documents/GitHub/policy-classifier/policy_scraping/policy_scraping/outputs/excel/full/"

def create_json_from_xlsx():
    df = pd.read_excel(XLSX_FILE)

    f_list = []

    for ind in df.index:
        try:
            dct = {
                'id':str(df.loc[ind,"id"]), 
                'name':df.loc[ind,"name"], 
                'engname':df.loc[ind,"engname"], 
                'level':df.loc[ind,"level"], 
                'country':df.loc[ind,"country"], 
                'nuts1name':str(df.loc[ind,"nuts1name"]), 
                'nuts2name':str(df.loc[ind,"nuts2name"]),
                'nuts3name':str(df.loc[ind,"nuts3name"]), 
                'localauth':str(df.loc[ind,"localauth"]), 
                'site':df.loc[ind,"site"], 
                'classif':df.loc[ind,"classif"], 
                'dates':df.loc[ind,"dates"], 
                'publisher':df.loc[ind,"publisher"],
                'excerpt':str(df.loc[ind,"excerpt"]), 
                'engexc':str(df.loc[ind,"engexc"]), 
                'abstract':str(df.loc[ind,"abstract"]), 
                'engabst':str(df.loc[ind,"engabst"]), 
                'lang':df.loc[ind,"lang"], 
                'link':df.loc[ind,"link"],
                'longpdf_link':str(df.loc[ind,"longpdf_link"]), 
                'shortpdf_link':str(df.loc[ind,"shortpdf_link"]),
                'hash_name': hashlib.sha1(str(df.loc[ind,"longpdf_link"]).encode()).hexdigest()
            }
            f_list.append(dct)
            #print(f"Succeeded in reading {df.loc[ind,'name']}")
        except Exception as e:  # In case the file is corrupted
            print(f"Could not read {df.loc[ind,'name']} due to {e}")

    with open(OUTPUT_FILE, 'w', encoding="utf-8") as outfile:
            json.dump(f_list, outfile, ensure_ascii=False, indent=4)

def add_file_xts():
    count=0
    for file in glob.glob(INPUT_FOLDER+"\\**\\*", recursive=True):
        if file[-4:] == ".pdf":
            
            pass
        else:
            old = file
            #print(old)
            new = file+".pdf"
            #print(new)
            count+=1
            os.rename(old, new)
    print(count)

        
def main():
    #create_json_from_xlsx()
    add_file_xts()  
    print("all done.")  

if __name__ =='__main__':
    main()