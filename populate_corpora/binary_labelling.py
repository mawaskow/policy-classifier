import json
import os
from os import listdir
from os.path import isfile, join
import random
import time
import datetime
import csv
import sys
import argparse
import tqdm
import textwrap

# Model libraries
from sentence_transformers import SentenceTransformer
import sentencepiece
from scipy.spatial import distance
import numpy as np

from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
QUERIES_DCT = {
    "Otorgamiento de estímulos crediticios por parte de el estado" : "Credit-México",
"Estos créditos podrían beneficiar a sistemas productivos asociados a la pequeña y mediana producción" : "Credit-Perú",
"Se asocia con créditos de enlace del Banco del Estado" : "Credit-Chile", 
"Acceso al programa de garantía crediticia para la actividad económica" : "Credit-Guatemala",
"El banco establecerá líneas de crédito para que el sistema financiero apoye la pequeña, mediana y microempresa" : "Credit-El Salvador",
"Dentro de los incentivos económicos se podrá crear un bono para retribuir a los propietarios por los bienes y servicios generados." : "Direct_payment-México",
"Acceso a los fondos forestales para el pago de actividad" : "Direct_payment-Perú",
"Se bonificará el 90% de los costos de repoblación para las primeras 15 hectáreas y de un 75% respecto las restantes" : "Direct_payment-Chile",
"El estado dará un incentivo que se pagará una sola vez a los propietarios forestales" : "Direct_payment-Guatemala",
"Incentivos en dinero para cubrir los costos directos e indirectos del establecimiento y manejo de areas de producción" : "Direct_payment-El Salvador",
"Toda persona física o moral que cause daños estará obligada a repararlo o compensarlo" : "Fine-México",
"Disminuir los riesgos para el inversionista implementando mecanismos de aseguramiento" : "Guarantee-México",
"Podrá garantizarse el cumplimiento de la actividad mediante fianza otorgada a favor del estado por cualquiera de las afianzadoras legalmente autorizadas." : "Guarantee-Guatemala",
"El sujeto de derecho podrá recibir insumos para la instalación y operación de infraestructuras para la actividad económica." : "Supplies-México",
"Se facilitará el soporte técnico a  través de la utilización de guías, manuales, protocolos, paquetes tecnológicos, procedimientos, entre otros." : "Supplies-Perú",
"Se concederán incentivos en especie para fomentar la actividad en forma de insumos" : "Supplies-El Salvador",
"Se otorgarán incentivos fiscales para la actividad primaria y también la actividad de transformación" : "Tax_deduction-México",
"De acuerdo con los lineamientos aprobados se concederá un 25% de descuento en el pago del derecho de aprovechamiento" : "Tax_deduction-Perú",
"Las bonificaciones percibidas o devengadas se considerarán como ingresos diferidos en el pasivo circulante y no se incluirán para el cálculo de la tasa adicional ni constituirán renta para ningún efecto legal hasta el momento en que se efectúe la explotación o venta" : "Tax_deduction-Chile",
"Los contratistas que suscriban contratos de exploración y/o explotación, quedan exentos de cualquier impuesto sobre los dividendos, participaciones y utilidades" : "Tax_deduction-Guatemala",
"Exención de los derechos e impuestos, incluyendo el Impuesto a la Transferencia de Bienes Muebles y a la Prestación de Servicios, en la importación de sus bienes, equipos y accesorios, maquinaria, vehículos, aeronaves o embarcaciones" : "Tax_deduction-El Salvador",
"Se facilitará formación Permanente Además del acompañamiento técnico, los sujetos de derecho participarán en un proceso permanente de formación a lo largo de todo el año, que les permita enriquecer sus habilidades y capacidades " : "Technical_assistance-México",
"Contribuir en la promoción para la gestión, a través de la capacitación, asesoramiento, asistencia técnica y educación de los usuarios" : "Technical_assistance-Perú",
"Asesoría prestada al usuario por un operador acreditado, conducente a elaborar, acompañar y apoyar la adecuada ejecución técnica en terreno de aquellas prácticas comprometidas en el Plan de Manejo" : "Technical_assistance-Chile",
"Para la ejecución de programas de capacitación, adiestramiento y otorgamiento de becas para la preparación de personal , así como para el desarrollo de tecnología en actividades directamente relacionadas con las operaciones objeto del contrato" : "Technical_assistance-Guatemala",
"Apoyo técnico y en formulación de proyectos y conexión con mercados" : "Technical_assistance-El Salvador"}

def labeled_sentences_from_dataset(dataset):
    sentence_tags_dict = {}
    for document in dataset.values():
        sentence_tags_dict.update(document['sentences'])
    return sentence_tags_dict

def create_sentence_embeddings(model, sentences_dict):
    embeddings = {}
    for sentence_id, sentence_map in sentences_dict.items():
        embeddings[sentence_id] = model.encode(sentence_map['text'].lower(), show_progress_bar=False)
    return embeddings
    
def sentence_dissimilarity_search(model, queries, sentence_embeddings, sentences, similarity_limit, results_limit, cuda, prog_bar):
    results = {}
    for query in tqdm.tqdm(queries):
        Ti = time.perf_counter()
        dissimilarities = get_distance(model, sentence_embeddings, sentences, query, similarity_limit, cuda, prog_bar)
        results[query] = dissimilarities[0:results_limit] #results[transformer][query] = similarities[0:results_limit]
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

def get_distance(model, sentence_emb, sentences_dict, query, dissimilarity_treshold, cuda, prog_bar):
    if cuda:
        query_embedding = model.encode(query.lower(), show_progress_bar=prog_bar, device='cuda')
    else:
        query_embedding = model.encode(query.lower(), show_progress_bar=prog_bar)
    highlights = []
    for sentence in sentences_dict.keys():
        try:
            sentence_embedding = sentence_emb[sentence]
            score = 1 - distance.cosine(sentence_embedding, query_embedding)
            if score < dissimilarity_treshold:
                highlights.append([sentence, score, sentences_dict[sentence]['text']])
        except KeyError as err:
            print(sentence)
            print(sentence_emb.keys())
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

# Adding the rank to each result
def add_rank(results_dictionary):
#     for model in results_dictionary:
    for keyword in results_dictionary:#[model]:
        i = 1
        for result in results_dictionary[keyword]:#[model][keyword]:
            result.insert(1, i)
            i += 1
    return results_dictionary

# For experiments 2 and 3 this function is to save results in separate csv files
def save_results_as_separate_csv(results_dictionary, queries_dictionary, path):
#     for model, value in results_dictionary.items():
    for exp_title, result in results_dictionary.items():#value.items():
        filename = queries_dictionary[exp_title]
        file = path + filename + ".csv"
        with open(file, 'w', newline='', encoding='utf-8') as f:
            write = csv.writer(f)
            write.writerows(result)
#             print(filename)
    
############################################################################
def run_embedder(sample=True, cuda=False, input_path=".", output_path="."):
    script_info = "Running "
    if sample:
        script_info += "sample"
    else:
        script_info += "all sentences"
    if cuda:
        script_info += " on GPU."
    else:
        script_info += " on CPU."
    print(script_info)
    
    policy_dict = {}
    onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]

    for f in onlyfiles:
        with open(join(input_path, f), 'r') as file:
            data = json.load(file)
        policy_dict = {**policy_dict, **data}

    sentences = labeled_sentences_from_dataset(policy_dict)

    if sample:
        #random.seed(9)
        sample_sentence_ids = random.sample(list(sentences), 10)
        sample_sentences = {}
        for s_id in sample_sentence_ids:
            sample_sentences.update({s_id: sentences[s_id]})
        sentences = sample_sentences

    Ti = time.perf_counter()

    transformer_name = 'xlm-r-bert-base-nli-stsb-mean-tokens'

    filename = "Embeddings.json"
    file = output_path + filename

    if cuda:
        model = SentenceTransformer(transformer_name, device="cuda")
    else:
        model = SentenceTransformer(transformer_name)
    
    print("Loaded model. Now creating sentence embeddings.")

    embs = create_sentence_embeddings(model, sentences)

    Tf = time.perf_counter()

    print(f"The building of a sentence embedding database in the current models has taken {Tf - Ti:0.2f}s.")

    with open(file, 'w+') as fp:
        json.dump(embs, fp, cls = NumpyArrayEncoder)

    return embs, sentences, model

def run_binary(embs, sentences, model, cuda=False, output_path=".",sim_thresh=0.2, res_lim=1000):
    prog_bar = False
    print("Now running queries.")
    queries = []
    for query in QUERIES_DCT:
        queries.append(query)

    #check_dictionary_values(queries_dict)

    results_dict = sentence_dissimilarity_search(model, queries, embs, sentences, sim_thresh, res_lim, cuda, prog_bar)

    file = output_path + "Pre_tagged_dissim.json"
    with open(file, 'w+') as fp:
        json.dump(results_dict, fp, indent=4)

def add_rank(file, query_dct, output_path):
    with open(file, "r") as f:
        results_ = json.load(f)
    results = add_rank(results_)
    # Save the results as separete csv files
    save_results_as_separate_csv(results, query_dct, output_path)

def convert_pretagged(input_path, output_path):
    with open(input_path, "r") as f:
        pre_lab = json.load(f)
    sentences = []
    for qry in tqdm.tqdm(list(pre_lab)):
        for sent_unit in pre_lab[qry]:
            # the format has the sentence as the last element in the sublist
            sentence = sent_unit[-1]
            sentences.append(sentence)
    with open(output_path, 'w+') as fp:
        json.dump(sentences, fp, indent=2)

def main():
    '''
    st = time.time()
    sample = False
    cuda = True
    input_path = "C:/Users/Allie/Documents/GitHub/policy-data-analyzer/tasks/text_preprocessing/output/new/"
    output_path = "C:/Users/Allie/Documents/GitHub/policy-data-analyzer/tasks/data_augmentation/output/EXP1/"
    if sample:
        sim_thresh = 0.2
    else:
        sim_thresh = 0.5
    ##############
    embs, sentences, model = run_embedder(sample, cuda, input_path, output_path)
    #run_queries(embs, sentences, model, cuda, output_path, sim_thresh=0.4, res_lim=1000)
    run_binary(embs, sentences, model, cuda, output_path, sim_thresh=0.05, res_lim=1000)
    ##############
    et = time.time()-st
    print("Time elapsed total:", et//60, "min and", round(et%60), "sec")
    '''
    input_path= "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\data_augmentation\\output\\EXP1\\Pre_tagged_dissim_0.05.json"
    output_path= "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\data_augmentation\\output\\EXP1\\Pre_tagged_dis_sentences.json"
    convert_pretagged(input_path, output_path)
    input_path= "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\data_augmentation\\output\\EXP1\\Pre_tagged_0.5.json"
    output_path= "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\data_augmentation\\output\\EXP1\\Pre_tagged_0.5_sentences.json"
    convert_pretagged(input_path, output_path)
    

if __name__ == '__main__':
    main()
    '''
    
    # cmd line arg
    # python binary_labeling.py -l 'spanish' -s -i "C:/Users/Ales/Documents/GitHub/policy-data-analyzer/tasks/text_preprocessing/output/new/" -o "C:/Users/Ales/Documents/GitHub/policy-data-analyzer/tasks/data_augmentation/output/sample/"

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