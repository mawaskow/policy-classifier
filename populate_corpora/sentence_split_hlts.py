from typing import Dict, List, Any, Set
import os
import nltk
#nltk.download('punkt')
import json
import argparse
import unidecode
import tqdm
# may need to move chdir import src.utils and Tokenizer back here

def preprocess_text(txt: str, remove_new_lines: bool = False) -> str:
    """
    Steps in the preprocessing of text:
        1. Remove HTML tags
        2. Replace URLS by a tag [URL]
        3. Replace new lines and tabs by normal spaces - sometimes sentences have new lines in the middle
        4. Remove excessive spaces (more than 1 occurrence)
        5. Parse emails and abreviations
    """
    txt = replace_links(remove_html_tags(txt)).strip()
    if remove_new_lines:
        txt = txt.replace("\n", " ").replace("\t", " ").strip()

    txt = remove_multiple_spaces(txt)
    txt = parse_emails(txt)
    txt = parse_acronyms(txt)

    new_txt = ""
    all_period_idx = set([indices.start() for indices in re.finditer("\.", txt)])

    for i, char in enumerate(txt):
        if i in all_period_idx:
            # Any char following a period that is NOT a space means that we should not add that period
            if i + 1 < len(txt) and txt[i + 1] != " ":
                continue

            # NOTE: Any char that is a number following a period will not count.
            # For enumerations, we're counting on docs being enumerated as "(a)" or "(ii)", and if not,
            # they will be separated by the "." after the number:
            # "Before bullet point. 3. Bullet point text" will just be "Before bullet point 3." and "Bullet point text" as the sentences
            if i + 2 < len(txt) and txt[i + 2].isnumeric():
                continue

            # If we wanted to have all numbered lists together, uncomment this, and comment out the previous condition
            # if i + 2 < len(txt) and not txt[i + 2].isalpha():
            #     continue

        new_txt += char

    return new_txt

def preprocess_spanish_text(txt: str, remove_new_lines: bool = False) -> str:
    return unidecode.unidecode(preprocess_text(txt, remove_new_lines))


def remove_short_sents(sents: List[str], min_num_words: int = 4) -> List[str]:
    """
    Remove sentences that are made of less than a given number of words. Default is 4
    """
    return [sent for sent in sents if len(sent.split()) >= min_num_words]


def get_nltk_sents(txt: str, tokenizer: nltk.PunktSentenceTokenizer, extra_abbreviations: Set[str] = None) -> List[str]:
    if extra_abbreviations is not None:
        tokenizer._params.abbrev_types.update(extra_abbreviations)

    return tokenizer.tokenize(txt)

###################################################
def keep_paragraph(input_path, output_path):
    abbrevs= None
    tokenizer = ES_TOKENIZER
    min_num_words = 5
    with open(input_path, "r", encoding="utf-8") as f:
        hlt_dct = json.load(f)
    for pdf in tqdm.tqdm(list(hlt_dct)):
        for pn in list(hlt_dct[pdf]):
            for sn in list(hlt_dct[pdf][pn]):
                sent = hlt_dct[pdf][pn][sn]["sentence"]
                preprocessed_text = preprocess_spanish_text(sent)
                sents = get_nltk_sents(preprocessed_text, tokenizer, abbrevs)
                postprocessed_sents = remove_short_sents(sents, min_num_words)
                hlt_dct[pdf][pn][sn]["sentence"] = " ".join(postprocessed_sents)
    with open(output_path, "w", encoding="utf-8") as fo:
        json.dump(hlt_dct, fo, ensure_ascii=False, indent=4)
    print("done")

def paragraph_to_sents(input_path, output_path):
    abbrevs= None
    tokenizer = ES_TOKENIZER
    min_num_words = 5
    with open(input_path, "r", encoding="utf-8") as f:
        hlt_dct = json.load(f)
    for pdf in tqdm.tqdm(list(hlt_dct)):
        for pn in list(hlt_dct[pdf]):
            new_page = {}
            for sn in list(hlt_dct[pdf][pn]):
                sent = hlt_dct[pdf][pn][sn]["sentence"]
                preprocessed_text = preprocess_spanish_text(sent)
                sents = get_nltk_sents(preprocessed_text, tokenizer, abbrevs)
                postprocessed_sents = remove_short_sents(sents, min_num_words)
                for i in range(len(postprocessed_sents)):
                    new_page[i] = {
                        "sentence": postprocessed_sents[i],
                        "label": hlt_dct[pdf][pn][sn]["label"]
                    }
            hlt_dct[pdf][pn]=new_page
    with open(output_path, "w", encoding="utf-8") as fo:
        json.dump(hlt_dct, fo, ensure_ascii=False, indent=4)
    print("done")

def main(input_path, output_path):
    #keep_paragraph(input_path, output_path)
    paragraph_to_sents(input_path, output_path)

if __name__ == '__main__':
    #TXT_PREP_DIR = ".."
    TXT_PREP_DIR = "C:\\Users\\Allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\text_preprocessing\\src"
    os.chdir(TXT_PREP_DIR)
    from utils import *

    #EN_TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")
    ES_TOKENIZER = nltk.data.load("tokenizers/punkt/spanish.pickle")

    input_path = "C:\\Users\\Allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\text_preprocessing\\output\\fixed_hlts.json"
    output_path = "C:\\Users\\Allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\text_preprocessing\\output\\sents_split.json"

    main(input_path, output_path)

    '''
    # cmd args
    # python ./src/sentence_split_local.py -i "C:\\Users\\Ales\\Documents\\GitHub\\policy-data-analyzer\\tasks\\extract_text\\output\\new\\pdf_files.json" -o "C:\\Users\\Ales\\Documents\\GitHub\\policy-data-analyzer\\tasks\\text_preprocessing\\output"

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', required=True,
                        help="Path to input file (pdf_files.json)")
    parser.add_argument('-o', '--output_dir', required=True,
                        help="Path to output folder where results will be stored.")

    args = parser.parse_args()

    main(args.input_path, args.output_dir)
    '''