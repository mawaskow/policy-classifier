import json
import nltk
#nltk.download('punkt')
from pdfs_to_jsons import pdfs_to_txt_dct
from json_cleaning import get_clean_text_dct

def main():
    pdf_dict = pdfs_to_txt_dct(INPUT_DIR)
    sent_dct = get_clean_text_dct(pdf_dict, EN_TOKENIZER)
    with open(OUTPUT_PTH+"/portal_pdfs.json", 'w', encoding="utf-8") as outfile:
        json.dump(sent_dct, outfile, ensure_ascii=False, indent=4)


if __name__ =="__main__":
    EN_TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")
    #
    #INPUT_DIR= "C:/Users/Allie/Documents/GitHub/policy-classifier/policy_scraping/outputs/full"
    INPUT_DIR= "C:/Users/Allie/Documents/PhD/IrishPoliciesMar24"
    OUTPUT_PTH = "C:/Users/Allie/Documents/GitHub/policy-classifier/populate_corpora/outputs"
    #
    main()
    
