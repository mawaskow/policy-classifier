import json

def format_sents_for_output(sents, doc_id):
    """
    Fxn from Firebanks-Quevedo repo
    """
    formatted_sents = {}
    for i, sent in enumerate(sents):
        formatted_sents.update({f"{doc_id}_sent_{i}": {"text": sent, "label": []}})
    return formatted_sents

def get_lists(messy):
    all_topics = {}
    for key in list(messy):
        all_topics.update(format_sents_for_output(messy[key], key))
    return all_topics

def crude_labels(dcts):
    for sent in list(dcts):
        if "general" in sent:
            dcts[sent]['label'].append("ment")
        #if "_gen_" in sent:
        #    dcts[sent]['label'].append("gen_pub")
        if "intention" in sent:
            dcts[sent]['label'].append("intent")
        if "plan" in sent:
            dcts[sent]['label'].append("plan")
        if "action" in sent:
            dcts[sent]['label'].append("action")
    return dcts

def dcts_to_lsts(dcts):
    txts = []
    labs = []
    for sent in list(dcts):
        txts.append(dcts[sent]['text'])
        labs.append(dcts[sent]['label'])
    return txts, labs

if __name__ == "__main__":
    infile_path = './ChatGPT/chatgpt_1x.json'
    outfile_path = './ChatGPT/chatgpt_1_labelledx.json'
    #
    with open(infile_path, 'r', encoding="UTF-8") as file:
        messy = json.load(file)
    #
    result = get_lists(messy)
    result = crude_labels(result)
    txts, labs = dcts_to_lsts(result)
    #
    #with open(outfile_path, 'w') as file:
    #    json.dump(result, file, indent=4)
    #
    ''''''
    snt_pth = './ChatGPT/sents_noactx.json'
    lab_pth = './ChatGPT/labs_noactx.json'
    with open(snt_pth, 'w') as file:
        json.dump(txts, file, indent=4)
    with open(lab_pth, 'w') as file:
        json.dump(labs, file, indent=4)
    
    print("done.")