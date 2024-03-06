#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import time
import argparse
import tqdm

def sentcheck_dups(input_path, out_path):
    '''
    input: file path (str) to json
    output: 
    '''
    with open(input_path, "r", encoding="utf-8") as f:
        pdf_ann = json.load(f)
    #traverse by filename
    for fn in list(pdf_ann):
        # then pagenumber
        for pn in list(pdf_ann[fn]):
            if len(list(pdf_ann[fn][pn]))>1:
                # create a new dictionary for the page
                new_pg = {}
                # traverse sentence keys by iterable
                for si in range(len(list(pdf_ann[fn][pn]))):
                    # if first sentence, add to new page (sentence and label)
                    if si== 0:
                        sn = list(pdf_ann[fn][pn])[si]
                        new_pg[sn] = pdf_ann[fn][pn][sn]
                    else:
                        # get previous and current sentence keys
                        pk = list(pdf_ann[fn][pn])[si-1]
                        ck = list(pdf_ann[fn][pn])[si]
                        # check if previous sentence is contained in present sentence
                        if pdf_ann[fn][pn][pk]["sentence"] in pdf_ann[fn][pn][ck]["sentence"]:
                            # if so, replace sentence and add label back
                            new_pg[ck] = {}
                            new_pg[ck]["sentence"] = pdf_ann[fn][pn][ck]["sentence"].replace(pdf_ann[fn][pn][pk]["sentence"], "")
                            new_pg[ck]["label"] = pdf_ann[fn][pn][ck]["label"]
                        else:
                            new_pg[ck] = pdf_ann[fn][pn][ck]
                pdf_ann[fn][pn] = new_pg
    with open(out_path, "w", encoding="utf-8") as fo:
        json.dump(pdf_ann, fo, ensure_ascii=False, indent=4)
             
    
def main(input, output):
    '''
    inputs: path of file to be converted (str), path of output (str)
    output: file with cleaned sentences
    Takes json file with raw highlight sentences and removes duplicate sentences
    '''
    sentcheck_dups(input, output)
    #
    '''
    test_dct= {
         "tst1": {
            "0": {
                "sentence": "cual postula. Asistencia técnica para la ejecución del Plan de Manejo: asesoría prestada al usuario por un Asistencia técnica para la ejecución del Plan Manejo: asesoría prestada al usuario por un operador acreditado, conducente a elaborar, Manejo: asesoría prestada al usuario por un operador acreditado, conducente a elaborar, acompañar y apoyar la adecuada ejecución técnica operador acreditado, conducente a elaborar, acompañar y apoyar la adecuada ejecución técnica en terreno de aquellas prácticas comprometidas acompañar y apoyar la adecuada ejecución técnica en terreno de aquellas prácticas comprometidas en el Plan de Manejo, sólo podrán postular en terreno de aquellas prácticas comprometidas en el Plan de Manejo, sólo podrán postular a esta asistencia, los pequeños productores Decreto 18, AGRICULTURA en el Plan de Manejo, sólo podrán postular a esta asistencia, los pequeños productores agrícolas. Decreto 18, AGRICULTURA Art. PRIMERO a esta asistencia, agrícolas. Asistencia AGRICULTURA Art. PRIMERO N° 1 D.O. 07.11.2015 2 bis. agrícolas. Asistencia técnica para la elaboración del Plan de Manejo: asesoría prestada al usuario por un Art. PRIMERO N° D.O. 07.11.2015 Decreto 18, Asistencia técnica para la elaboración del Plan de Manejo: asesoría prestada al usuario por un operador acreditado para la confección de un plan D.O. 07.11.2015 Decreto 18, AGRICULTURA de Manejo: asesoría prestada al usuario por un operador acreditado para la confección de un plan de manejo, la que debe contener, además de la Decreto 18, AGRICULTURA Art. PRIMERO operador acreditado para la confección de un de manejo, la que debe contener, además de la documentación exigida, una propuesta de las AGRICULTURA Art. PRIMERO N° 2 D.O. 07.11.2015 de manejo, la que debe contener, además de documentación exigida, una propuesta de las prácticas a postular, señalando la descripción Art. PRIMERO N° D.O. 07.11.2015 documentación exigida, una propuesta de las prácticas a postular, señalando la descripción pormenorizada de ellas y la ubicación de los prácticas a postular, señalando la descripción pormenorizada de ellas y la ubicación de los potreros potreros potreros a intervenir, sólo podrán postular a esta asistencia, a intervenir, asistencia, los asistencia, los pequeños productores agrícolas. Bosque: sitio poblado con formaciones",
                "label": [
                    "Technical assistance"
                ]
            },
            "1": {
                "sentence": "cual postula. Asistencia técnica para la ejecución del Plan de Manejo: asesoría prestada al usuario por un Asistencia técnica para la ejecución del Plan Manejo: asesoría prestada al usuario por un operador acreditado, conducente a elaborar, Manejo: asesoría prestada al usuario por un operador acreditado, conducente a elaborar, acompañar y apoyar la adecuada ejecución técnica operador acreditado, conducente a elaborar, acompañar y apoyar la adecuada ejecución técnica en terreno de aquellas prácticas comprometidas acompañar y apoyar la adecuada ejecución técnica en terreno de aquellas prácticas comprometidas en el Plan de Manejo, sólo podrán postular en terreno de aquellas prácticas comprometidas en el Plan de Manejo, sólo podrán postular a esta asistencia, los pequeños productores Decreto 18, AGRICULTURA en el Plan de Manejo, sólo podrán postular a esta asistencia, los pequeños productores agrícolas. Decreto 18, AGRICULTURA Art. PRIMERO a esta asistencia, agrícolas. Asistencia AGRICULTURA Art. PRIMERO N° 1 D.O. 07.11.2015 2 bis. agrícolas. Asistencia técnica para la elaboración del Plan de Manejo: asesoría prestada al usuario por un Art. PRIMERO N° D.O. 07.11.2015 Decreto 18, Asistencia técnica para la elaboración del Plan de Manejo: asesoría prestada al usuario por un operador acreditado para la confección de un plan D.O. 07.11.2015 Decreto 18, AGRICULTURA de Manejo: asesoría prestada al usuario por un operador acreditado para la confección de un plan de manejo, la que debe contener, además de la Decreto 18, AGRICULTURA Art. PRIMERO operador acreditado para la confección de un de manejo, la que debe contener, además de la documentación exigida, una propuesta de las AGRICULTURA Art. PRIMERO N° 2 D.O. 07.11.2015 de manejo, la que debe contener, además de documentación exigida, una propuesta de las prácticas a postular, señalando la descripción Art. PRIMERO N° D.O. 07.11.2015 documentación exigida, una propuesta de las prácticas a postular, señalando la descripción pormenorizada de ellas y la ubicación de los prácticas a postular, señalando la descripción pormenorizada de ellas y la ubicación de los potreros potreros potreros a intervenir, sólo podrán postular a esta asistencia, a intervenir, asistencia, los asistencia, los pequeños productores agrícolas. Bosque: sitio poblado con formaciones Nº 20.283. Capacitación: actividad destinada a transmitir conocimientos técnicos o administrativos necesarios Capacitación: actividad destinada a transmitir conocimientos técnicos o administrativos necesarios para el mejor logro de los objetivos del Programa, implementada por el Instituto de Desarrollo para el mejor logro de los objetivos del Programa, implementada por el Instituto de Desarrollo Agropecuario, en adelante indistintamente INDAP, implementada por el Instituto de Desarrollo Agropecuario, en adelante indistintamente INDAP, o por el Servicio Agrícola y Ganadero, en adelante Agropecuario, en adelante indistintamente INDAP, o por el Servicio Agrícola y Ganadero, en adelante indistintamente SAG, o por terceros a su nombre o por el Servicio Agrícola y Ganadero, en adelante indistintamente SAG, o por terceros a su nombre y con su autorización, en materias propias indistintamente SAG, o por terceros a su nombre y con su autorización, en materias propias del Programa. y con su autorización, del Programa. Costo neto: valor",
                "label": [
                    "Technical assistance"
                ]
            }
        },
        "test2": {
            "0": {
                "sentence": "Se procederá a pagar el incentivo por corresponda, de acuerdo a la Tabla Anual Artículo 43.- Se procederá a pagar el incentivo por INDAP o SAG, según corresponda, de acuerdo a la Tabla Anual de Costos vigente al momento de la recepción de la INDAP o SAG, según corresponda, de acuerdo a la Tabla de Costos vigente al momento de la recepción de la postulación. de Costos vigente postulación. En el caso postulación. En el caso de las personas jurídicas beneficiarias de bonificación, el pago sólo procederá siempre que En el caso de las personas jurídicas beneficiarias esta bonificación, el pago sólo procederá siempre que ellas se encuentren inscritas en el Registro de Personas esta bonificación, el pago sólo procederá siempre que ellas se encuentren inscritas en el Registro de Personas Jurídicas Receptoras de Fondos Públicos, conforme a lo ellas se encuentren inscritas en el Registro de Personas Jurídicas Receptoras de Fondos Públicos, conforme a lo establecido en la ley Nº 19.862 (www.registros19862.cl). Jurídicas Receptoras de Fondos establecido en la ley Nº 19.862 Mediante un mandato simple, el",
                "label": [
                    "Direct payment"
                ]
            },
            "1": {
                "sentence": "Se procederá a pagar el incentivo por corresponda, de acuerdo a la Tabla Anual Artículo 43.- Se procederá a pagar el incentivo por INDAP o SAG, según corresponda, de acuerdo a la Tabla Anual de Costos vigente al momento de la recepción de la INDAP o SAG, según corresponda, de acuerdo a la Tabla de Costos vigente al momento de la recepción de la postulación. de Costos vigente postulación. En el caso postulación. En el caso de las personas jurídicas beneficiarias de bonificación, el pago sólo procederá siempre que En el caso de las personas jurídicas beneficiarias esta bonificación, el pago sólo procederá siempre que ellas se encuentren inscritas en el Registro de Personas esta bonificación, el pago sólo procederá siempre que ellas se encuentren inscritas en el Registro de Personas Jurídicas Receptoras de Fondos Públicos, conforme a lo ellas se encuentren inscritas en el Registro de Personas Jurídicas Receptoras de Fondos Públicos, conforme a lo establecido en la ley Nº 19.862 (www.registros19862.cl). Jurídicas Receptoras de Fondos establecido en la ley Nº 19.862 Mediante un mandato simple, el Los incentivos a que se refiere la ley con los establecidos en otros cuerpos Artículo 45.- Los incentivos a que se refiere la ley serán compatibles con los establecidos en otros cuerpos legales o reglamentarios sobre fomento a la actividad serán compatibles con los establecidos en otros cuerpos legales o reglamentarios sobre fomento a la actividad agropecuaria, forestal y ambiental, pero el conjunto legales o reglamentarios sobre fomento a la actividad agropecuaria, forestal y ambiental, pero el conjunto de los que obtenga un mismo productor respecto de un mismo predio agropecuaria, forestal y ambiental, pero el conjunto de los que obtenga un mismo productor respecto de un mismo predio y de una misma práctica, no podrá exceder el 100% de los que obtenga un mismo productor respecto de un mismo predio de una misma práctica, no podrá exceder el 100% de los costos de las labores o insumos bonificados. de una misma práctica, no podrá exceder el 100% costos de las labores o insumos bonificados.",
                "label": [
                    "Direct payment"
                ]
            }
        },

    }

    for i in list(test_dct):
        for j in list(test_dct[i]):
            if j!= list(test_dct[i])[0]:
                if test_dct[i][str(int(j)-1)]["sentence"] in test_dct[i][j]["sentence"]:
                    print("\n\n\n", test_dct[i][str(int(j)-1)]["sentence"])
                    print("\n", test_dct[i][j]["sentence"].replace(test_dct[i][str(int(j)-1)]["sentence"], ""))
    '''
    #
    print("done")


if __name__ == '__main__':
    input_path = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\text_preprocessing\\output\\fixed_empty.json"
    output_path = "C:\\Users\\allie\\Documents\\GitHub\\policy-data-analyzer\\tasks\\text_preprocessing\\output\\fixed_hlts.json"
    main(input_path, output_path)