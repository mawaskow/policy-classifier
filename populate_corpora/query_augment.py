'''
Adapted from old repo's /tasks/data_augmentation/assisted_labelling.py
'''
import json
import os
from os import listdir
from os.path import isfile, join
import random
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import numpy as np
from collections import Counter
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def create_sentence_embeddings(model, sentences):
    embeddings = []
    for sentence in sentences:
        embeddings.append(model.encode(sentence.lower(), show_progress_bar=False))
    return embeddings
    
def sentence_similarity_search(model, queries, embeddings, sentences, similarity_limit, results_limit, cuda, prog_bar):
    results = {}
    for query in tqdm(queries):
        Ti = time.perf_counter()
        similarities = get_distance(model, embeddings, sentences, query, similarity_limit, cuda, prog_bar)
        results[query] = similarities[0:results_limit] #results[transformer][query] = similarities[0:results_limit]
        Tf = time.perf_counter()
        print(f"Similarity search for query '{query}' has been done in {Tf - Ti:0.2f}s.")
    return results

# This function helps debugging misspelling in the values of the dictionary
def check_dictionary_values(dictionary):
    check_country = {}
    check_incentive = {}
    for key, value in dictionary.items():
        incentive, country = value.split("-")
        check_incentive[incentive] = 0
        check_country[country] = 0
    print(check_incentive)
    print(check_country)

def get_distance(model, embeddings, sentences, query, similarity_treshold, cuda, prog_bar):
    if cuda:
        query_embedding = model.encode(query.lower(), show_progress_bar=prog_bar, device='cuda')
    else:
        query_embedding = model.encode(query.lower(), show_progress_bar=prog_bar)
    highlights = []
    for i in range(len(sentences)):
        try:
            sentence_embedding = embeddings[i]
            score = 1 - distance.cosine(sentence_embedding, query_embedding)
            if score > similarity_treshold:
                highlights.append([i, score, sentences[i]])
        except KeyError as err:
            print(sentences[i])
            print(embeddings[i])
            print(err)
    highlights = sorted(highlights, key = lambda x : x[1], reverse = True)
    return highlights

# To show the contents of the results dict, particularly, the length of the first element and its contents
def show_results(results_dictionary):
    i = 0
    for key1 in results_dictionary:
        for key2 in results_dictionary[key1]:
            if i == 0:
                print(len(results_dictionary[key1][key2]))
                print(results_dictionary[key1][key2])
            i += 1

############################################################################

QUERIES_DCT = {
  "This scheme gives farmers greater access to financial loans and encourages financial planning." : "Credit",
  "National initiatives, such as the Future Growth Loan Scheme, supports strategic long-term capital investment by providing competitively priced loan instruments under favourable terms." : "Credit",
  "Harvest and production insurance that contributes to safeguarding producers' incomes where there are losses as a consequence of natural disasters, adverse climatic events, diseases or pest infestations while ensuring that beneficiaries take necessary risk prevention measures." : "Credit",
  "The Department of Agriculture Food and the Marine has funded a number of loan schemes which provide access to finance for famers in Ireland, enabling them to maintain liquidity and ensure they can take investment decisions tailored to their enterprise" : "Credit",
  "In cases where a loan is used to finance or top up a mutual fund no distinction is made between the basic capital and loans taken out in respect of the replenishment of the fund following compensate to growers." : "Credit",
  "The Scheme is supporting generational renewal on Irish farms by allowing young farmers to avail of a higher grant rate of 60% (with a standard grant rate of 40% available to all other applicants)." : "Direct_payment",
  "This programme incorporated extra payments on top of the basic REPS premium for farmers who undertook additional environmentally friendly farming practices." : "Direct_payment",
  "Forestry Programme 2014 -2020 providing grants and / or annual premiums for establishment, development and reconstitution of forests, woodland improvement, native woodland conservation." : "Direct_payment",
  "In addition to providing a basic income support to primary producers and achieving a higher level of environmental ambition, Pillar I (direct payments) interventions are aimed at achieving a fairer approach to the distribution of payments in Ireland." : "Direct_payment",
  "Decision to set an amount of direct payments not higher than EUR 5 000 under which farmers shall in any event be considered as ‘active farmers.’" : "Direct_payment",
  "Landowners found burning illegally could face fines, imprisonment and Single Farm Payment penalties, where applicable." : "Fine",
  "In the absence of abatement strategies, ammonia emissions are forecast to increase which may result either in substantial fines or the imposition of a de-facto quota based on emission levels." : "Fine",
  "If an offence is committed by a public body, and is committed with the consent of, or is attributable to the neglect on the part of a director, manager or other officer of the public body, that person will also be liable for prosecution; on conviction, fines up to €250,000 or imprisonment for up to 2 years, or both, may be imposed 125 Per s ection 14A(6) of the Act." : "Fine",
  "In addition, a fine will apply which will be calculated on the difference between the area declared and the area determined." : "Fine",
  "Where trees have been—(a) felled or otherwise removed without a licence under section 7,(b) felled under a licence and, either at the time of such felling or subsequently, a condition of the licence is contravened, or(c) in the opinion of the Minister, seriously damaged, the Minister may issue a replanting order in respect of the owner requiring him or her to replant or to fulfil any or all of the conditions that attached to the licence (or, in a case in which no licence was granted, any or all of the conditions that would, in the opinion of the Minister, have been attached to a licence had such been granted) in accordance with the provisions of the orderSections 27-29 detail offences and corresponding penalties, from fixed penalties to (on conviction) substantial fines and imprisonment." : "Fine",
  "Similarly, the On-Farm Capital Investments Scheme has provisions for investments in equipment that will allow farmers to reduce the amount of Green House Gas emissions that they produce during their agricultural practices." : "Supplies",
  "The intervention also supports investments that allow farmers to acquire technologies and equipment that increases their efficiencies and climate adaptation potential thus addressing Obj4N4 and Obj4N5." : "Supplies",
  "Grants are provided for farmers wishing to invest in productive technologies and or equipment." : "Supplies",
  "Support provided under this scheme will directly address Obj5N1 and Obj5N2 by providing an incentive to farmers to invest in machinery and equipment that better protects air and water quality." : "Supplies",
  "The On Farm Capital Investment Scheme also addresses Obj9N1 by providing a higher grant rate of support for investments in organic farming materials/equipment, at a higher rate of 60% in comparison to the rate of 40% for general investments." : "Supplies",
  "To complement the EXEED programme, the tax code provides for accelerated capital allowances (ACAs) for energy efficient equipment supporting the reduction of energy use in the workplace and the awareness of energy efficiency standards in appliances." : "Tax_deduction",
  "A tax incentive for companies paying corporation tax is also in place in the form of accelerated capital allowance for energy efficient equipment." : "Tax_deduction",
  "The Accelerated Capital Allowance (ACA) is a tax credit that encourages the purchase of energy -efficient goods" : "Tax_deduction",
  "These include the granting of an enhanced 50% stock tax relief to members of registered farm partnerships; the recognition of such arrangements in the calculation of payments under the Pillar I and Pillar II Schemes; and the introduction of a Support for Collaborative Farming Grant Scheme for brand new farm partnerships." : "Tax_deduction",
  "We are committed to further developing a taxation framework, which plays its full part in incentivising, along with other available policy levers, the necessary actions to reduce our emissions" : "Tax_deduction",
  "The Knowledge Transfer (KT) initiative is a significant investment in high quality training and upskilling of farmers so that they are equipped to deal with the range of challenges and opportunities arising in the agri-food sector." : "Technical_assistance",
  "The associated training will educate farmers on how to appropriately implement the actions of the scheme; thereby equipping them with the knowledge and skills necessary to optimise delivery and continue the ongoing management of the commitments undertaken; as well as to facilitate the implementation of high welfare practices." : "Technical_assistance",
  "This scheme has two measures: • provides financial support towards the professional costs, such as legal, taxation and advisory, incurred during the establishment of a Registered Farm Partnership." : "Technical_assistance",
  "LEADER may provide support rates greater than 65% in accordance with Article 73(4) (c)(ii) where investments include basic services in rural areas and infrastructure in agriculture and forestry , as determined by Member States" : "Technical_assistance",
  "It also assists and supports the delivery of capacity building and training programmes with the aim of equipping decision makers with the capability and confidence to analyse, plan for and respond to the risks and opportunities that a changing climate presents." : "Technical_assistance"
  }

def run_embedder(sample=True, dev=None, data=[], unique=True):
    if dev:
        print("Running on GPU")
    else:
        print("Running on CPU")
    if unique:
        sentences = list(set(data))
    else:
        sentences = data
    if sample:
        sentences = random.sample(sentences, 10)
    Ti = time.perf_counter()
    transformer_name = 'xlm-r-bert-base-nli-stsb-mean-tokens'
    model = SentenceTransformer(transformer_name, device=dev)
    print("Loaded model. Now creating sentence embeddings.")
    embs = create_sentence_embeddings(model, sentences)
    Tf = time.perf_counter()
    print(f"The building of a sentence embedding database in the current models has taken {Tf - Ti:0.2f}s.")
    return embs, sentences, model

def run_queries(embs, sentences, model, qry_dct=QUERIES_DCT, dev=None, sim_thresh=0.6, res_lim=1000):
    prog_bar = False
    print("Now running queries.")
    queries = []
    for query in qry_dct:
        queries.append(query)
    results_dict = sentence_similarity_search(model, queries, embs, sentences, sim_thresh, res_lim, dev, prog_bar)
    return results_dict

def consolidate_sents(pre_lab, queries_dct=QUERIES_DCT):
    '''
    Takes dict in the form of 
    dict[query] = [[s_id1, sim_score1, sentence1], [s_id2, sim_score2, sentence2]...]
    and returns dict in the form of
    dict[label] = [sentence1, sentence2,...]
    '''
    new_dct = {}
    qdct = queries_dct.copy()
    for lbl in list(set(qdct.values())):
        new_dct[lbl]=[]
    for query in list(pre_lab):
        for entry in pre_lab[query]:
            new_dct[qdct[query]].append(entry[-1])
    return new_dct

def crossref_sents(lab_dct, thresh=4):
    '''
    Takes dict in the form of 
    dict[label] = [sentence1, sentence2,...]
    and returns dict in the form of
    dict[label] = [sentence1, sentence2,...]
    for all sentences that occurred at least 4 times
    [i.e. at least four of the queries for each label returned that sentence]
    '''
    from collections import Counter
    new_dct = {key: [] for key in list(lab_dct)}
    for lbl in list(lab_dct):
        c = Counter(lab_dct[lbl])
        for sent in list(c):
            if c[sent] >= thresh:
                new_dct[lbl].append(sent)
    return new_dct

def main():
    cwd = os.getcwd()
    st = time.time()
    sample = False
    dev = 'cuda'
    input_path = cwd+"/populate_corpora/outputs/full_text_sents.json"
    output_path = cwd+"/populate_corpora/outputs/"
    ##############
    with open(input_path,"r", encoding="utf-8") as f:
        sentences = json.load(f)
    embs, s_sentences, model = run_embedder(sample, dev, sentences)
    pret_dct = run_queries(embs, s_sentences, model, QUERIES_DCT, dev, sim_thresh=0.5, res_lim=1000)

    ##############
    et = time.time()-st
    print("Time elapsed total:", et//60, "min and", round(et%60), "sec")


if __name__ == '__main__':
    main()
    '''
    # cmd line arg
    # python assisted_labeling.py -l 'spanish' -s -i "C:/Users/Ales/Documents/GitHub/policy-data-analyzer/tasks/text_preprocessing/output/new/" -o "C:/Users/Ales/Documents/GitHub/policy-data-analyzer/tasks/data_augmentation/output/sample/"

    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--lang', required=True,
                        help="Language for sentence preprocessing/splitting. Current options are: 'spanish'")
    parser.add_argument('-s', '--sample', required=False, default=False, action='store_true',
                        help="Run sample of sentences instead of all sentences")
    parser.add_argument('-c', '--cuda', required=False, default=False, action='store_true',
                        help="Use cuda to run (if cuda-enabled GPU)")
    parser.add_argument('-i', '--input_path', required=True,
                        help="Input path for sentence split jsons.")
    parser.add_argument('-o', '--output_path', required=True,
                        help="Output path for result jsons.")
    parser.add_argument('-t', '--thresh', required=False, default=0.2,
                        help="Similarity threshold for sentences.")
    parser.add_argument('-r', '--lim', required=False, default=1000,
                        help="Results limit for sentence search.")

    args = parser.parse_args()

    main(args.lang, args.sample, args.cuda, args.input_path, args.output_path, float(args.thresh), int(args.lim))
    '''