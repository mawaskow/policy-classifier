import json
import time
import argparse
import tqdm

ann_cls_lst = [
    "Direct payment",
    "Credit",
    "Tax deduction",
    "Technical assistance",
    "Supplies",
    "Fine"
]

ann_cls_dct = {
    "direct": "Direct payment",
    "direct payment": "Direct payment",
    "credit": "Credit",
    "tax deduction": "Tax deduction",
    "technical assistance": "Technical assistance",
    "supplies": "Supplies",
    "fine": "Fine",
    "techical assistance": "Technical assistance",
    "tax credit": "Tax deduction",
    "pes": "Direct payment",
    "technical support": "Technical assistance",
    "direct payments": "Direct payment",
    "fines":"Fine"
}

def merge_label(input_string):
    '''
    input: label (str)
    returns: cleaned label (list)
    Goes through the raw label and converts it to
    elements a list if it can be converted into
    a known label
    '''
    label_lst = []
    try:
        input_string = input_string.split(",")
        for i in input_string:
            i = i.strip().lower()
            if i in ann_cls_dct.keys():
                label_lst.append(ann_cls_dct[i])
            elif i[:15] in ann_cls_dct.keys():
                label_lst.append(ann_cls_dct[i[:15]])
            else:
                i=i.replace("(", "")
                i=i.replace(")","")
                i=i.split(" ")
                for j in i:
                    if j=='pes':
                        label_lst.append("Direct payment")
        return label_lst
    except Exception as e:
        return [e]

def label_show_str(input_path):
    '''
    input: file path (str) to json
    output: prints foreign label,
            prints #foreign labels out of #labels
    Shows the labels in a file that dont match the known labels
    when the file labels are strings (not list elements)
    '''
    with open(input_path, "r", encoding="utf-8") as f:
        pdf_ann = json.load(f)
    i=0
    j=0
    #traverse by filename
    for fn in pdf_ann.keys():
        # then pagenumber
        for pn in pdf_ann[fn].keys():
            #then sentence in page
            for sn in pdf_ann[fn][pn].keys():
                j+=1
                label = pdf_ann[fn][pn][sn]['label']
                if label not in ann_cls_lst:
                    print(label)
                    i=i+1
    print(i, j)

def label_show_lst(input_path):
    '''
    input: file path (str) to json
    output: prints foreign label,
            prints #foreign labels out of #labels
    Shows the labels in a file that arent known labels
    when the file labels are list elements
    '''
    with open(input_path, "r", encoding="utf-8") as f:
        pdf_ann = json.load(f)
    i=0
    j=0
    #traverse by filename
    for fn in pdf_ann.keys():
        # then pagenumber
        for pn in pdf_ann[fn].keys():
            #then sentence in page
            for sn in pdf_ann[fn][pn].keys():
                j+=1
                label = pdf_ann[fn][pn][sn]['label']
                for lb in range(len(label)):
                    if label[lb] not in ann_cls_lst:
                        print(label[lb])
                        i=i+1
    print(i, j)

def clean_labels(input, output):
    '''
    inputs: path of file to be converted (str), path of output (str)
    output: file with clean/uniform labels
    Takes json file with raw annotations and converts them into lists of uniform labels (in new file)
    '''
    #label_show_str(input)
    '''
    # testing labels
    input_strings = ["Direct payment (PES)", "Forest, Agriculture (Crop)", "Direct payment (PES), Credit,",
                     "Credit, Technical assistance", " a policy in itself but a very important criticism of the inadequacies of previous financial incentive programs, namely PRODEFOR, PROCYMAF, PRODEPLAN. This argues that the programs lack long-term security due to dependency on budgets and lack of private investment. It also states that the previous programs were inflexible to different regional situations.",
                     "Fines", "Direct payments (PES), Technical assistance", "Other (Environmental education)",
                     "Unknown, Technical assistance", "PES, credit, technical assistance"]
    for i in input_strings:
        print(i, "BECOMES", merge_label(i))
    '''
    with open(input,"r", encoding="utf-8") as f:
        mess = json.load(f)
    for pdf in tqdm.tqdm(mess.keys()):
        for pg in mess[pdf].keys():
            for snt in mess[pdf][pg].keys():
                mess[pdf][pg][snt]["label"] = merge_label(mess[pdf][pg][snt]["label"])
    with open(output, "w", encoding="utf-8") as fo:
        json.dump(mess, fo, ensure_ascii=False, indent=4)
    label_show_lst(output)
    print("done")

def remove_empty(input, output):
    with open(input,"r", encoding="utf-8") as f:
        empties = json.load(f)
    for pdf in list(empties):
        for pg in list(empties[pdf]):
            for snt in list(empties[pdf][pg]):
                if len(empties[pdf][pg][snt]["label"]) == 0:
                    empties[pdf][pg].pop(snt)
            if len(empties[pdf][pg].keys()) == 0:
                empties[pdf].pop(pg)
        if len(empties[pdf].keys()) == 0:
            empties.pop(pdf)
    with open(output, "w", encoding="utf-8") as fo:
        json.dump(empties, fo, ensure_ascii=False, indent=4)

def main():
    raw_labels = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\extract_text\\output\\pdf_extract.json"
    fixed_labels = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\text_preprocessing\\output\\fixed_labels.json"
    clean_labels(raw_labels, fixed_labels)
    output_path = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\text_preprocessing\\output\\fixed_empty.json"
    remove_empty(fixed_labels, output_path)

if __name__ == '__main__':
    main()