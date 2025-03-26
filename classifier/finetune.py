from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, cohen_kappa_score
from sentence_transformers import SentencesDataset, SentenceTransformer, InputExample
import torch
from torch.utils.data import DataLoader
from torch import device, Tensor
import torch.nn as nn
from typing import Iterable, Dict
from sentence_transformers import losses
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import os
import json
import time
import random
from rapidfuzz import fuzz
import numpy as np
import math
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate
from datasets import Dataset, DatasetDict
from collections import Counter

cwd = os.getcwd()
output_dir = cwd+"/outputs/automodels_nofreeze"
input_dir = cwd+"/inputs"

from run_classifiers import group_duplicates, remove_duplicates, dcno_to_sentlab, gen_bn_sentlab, gen_mc_sentlab

def finetune_automodel(datasetdct, int2label, label2int, mode, model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1", dev='cuda', rstate=9, ts=0.20, oom=False):
    '''
    
    '''
    epochs = 10
    start = time.time()
    num_lbs = len(list(int2label))
    print(f'\nLoading model {model_name}\n')
    print("Tokenizing")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True).to(dev)
    tokenized_ds = datasetdct.map(preprocess_function, batched=True)
    # dont need collator if using Trainer
    #clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    #clf_metrics = evaluate.combine(["accuracy", "f1"])
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    metric_log = []
    def calc_metrics(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=1)
        #metrics = clf_metrics.compute(predictions=predictions, references=labels)
        metrics = {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": f1.compute(predictions=predictions, references=labels, average="weighted" if mode=="mc" else "binary")["f1"]
        }
        metric_log.append(metrics)
        return metrics
    print("Loading model")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_lbs,id2label=int2label, label2id=label2int).to(dev)
    ## freeze layers then unfreeze "pooler" layers
    #for name, param in model.base_model.named_parameters():
    #    param.requires_grad = False
    #for name, param in model.base_model.named_parameters():
    #    if "pooler" in name:
    #        param.requires_grad = True
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        seed=rstate,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        processing_class=tokenizer,
        compute_metrics=calc_metrics,
        #optimizers
        #optimizer_cls_and_kwargs
        #compute_loss_func=
    )
    print("Training")
    trainer.train()
    train_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    eval_losses = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
    print("Saving")
    trainer.save_model(output_dir+f"/{model_name.split('/')[-1]}_{mode}_e{epochs}_r{rstate}.pt")
    #
    predictions = trainer.predict(tokenized_ds["holdout"])
    # Extract the logits and labels from the predictions object
    logits = predictions.predictions
    labels = predictions.label_ids
    # Use your compute_metrics function
    metric_log.append({"train_loss":train_losses})
    metric_log.append({"eval_loss": eval_losses})
    metrics = calc_metrics((logits, labels))
    with open(output_dir+f"/{model_name.split('/')[-1]}_{mode}_e{epochs}_r{rstate}.pt/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metric_log, f, ensure_ascii=False, indent=4)
    end = time.time()
    print(metrics)
    print(f"\nSaved {model_name.split('/')[-1]}_{mode}_e{epochs}_r{rstate}.")
    print(f'\nDone in {round((end-start)/60,2)} min')

def create_labelintdcts(bn_labels, mc_labels, save = False):
    print(Counter(bn_labels)) 
    print(Counter(mc_labels))
    bn_int2label=dict(zip(range(len(set(bn_labels))), set(bn_labels)))
    bn_label2int = dict(zip(set(bn_labels), range(len(set(bn_labels)))))
    mc_int2label=dict(zip(range(len(set(mc_labels))), set(mc_labels)))
    mc_label2int = dict(zip(set(mc_labels), range(len(set(mc_labels)))))
    if save:
        with open(output_dir+f"/bn_int2label.json", "w", encoding="utf-8") as f:
            json.dump(bn_int2label, f, ensure_ascii=False, indent=4)
        with open(output_dir+f"/bn_label2int.json", "w", encoding="utf-8") as f:
            json.dump(bn_label2int, f, ensure_ascii=False, indent=4)
        with open(output_dir+f"/mc_int2label.json", "w", encoding="utf-8") as f:
            json.dump(mc_int2label, f, ensure_ascii=False, indent=4)
        with open(output_dir+f"/mc_label2int.json", "w", encoding="utf-8") as f:
            json.dump(mc_label2int, f, ensure_ascii=False, indent=4)
    int2label_dct = {
        "bn": bn_int2label,
        "mc": mc_int2label
    }
    label2int_dct = {
        "bn": bn_label2int,
        "mc": mc_label2int
    }
    return int2label_dct, label2int_dct

def load_labelintdcts():
    int2label_dct = {
        "bn": {
            0: "incentive",
            1: "non-incentive"
        },
        "mc":{
            0: "Fine",
            1: "Supplies",
            2: "Technical_assistance",
            3: "Tax_deduction",
            4: "Credit",
            5: "Direct_payment"
        }
    }
    label2int_dct = {
        "bn": {
            "incentive": 0,
            "non-incentive": 1
        },
        "mc":{
            "Fine": 0,
            "Supplies": 1,
            "Technical_assistance": 2,
            "Tax_deduction": 3,
            "Credit": 4,
            "Direct_payment": 5
        }
    }
    return int2label_dct, label2int_dct

def create_dsdict(sentences, labels, label2int_dct, amt=range(10), save=False):
    bn_sents, bn_labels = gen_bn_sentlab(sentences, labels, sanity_check=False)
    mc_sents, mc_labels = gen_mc_sentlab(sentences, labels, sanity_check=False)
    for e in amt:
        print(f"\nRound {e}\n")
        bn_ft_sents, bn_ho_sents, bn_ft_labels, bn_ho_labels = train_test_split(bn_sents, bn_labels, stratify=bn_labels, test_size=0.2, random_state=e)
        mc_ft_sents, mc_ho_sents, mc_ft_labels, mc_ho_labels = train_test_split(mc_sents, mc_labels, stratify=mc_labels, test_size=0.3, random_state=e)
        #
        bn_train_sents, bn_test_sents, bn_train_labels, bn_test_labels = train_test_split(bn_ft_sents, bn_ft_labels, stratify=bn_ft_labels, test_size=0.2, random_state=e)
        mc_train_sents, mc_test_sents, mc_train_labels, mc_test_labels = train_test_split(mc_ft_sents, mc_ft_labels, stratify=mc_ft_labels, test_size=0.3, random_state=e)
        #
        bn_ds = DatasetDict({
            "train": Dataset.from_list([{"text":text,"label":label2int_dct["bn"][label]} for text, label in zip(bn_train_sents, bn_train_labels)]),
            "test": Dataset.from_list([{"text":text,"label":label2int_dct["bn"][label]} for text, label in zip(bn_test_sents, bn_test_labels)]),
            "holdout": Dataset.from_list([{"text":text,"label":label2int_dct["bn"][label]} for text, label in zip(bn_ho_sents, bn_ho_labels)])
        })
        mc_ds = DatasetDict({
            "train": Dataset.from_list([{"text":text,"label":label2int_dct["mc"][label]} for text, label in zip(mc_train_sents, mc_train_labels)]),
            "test": Dataset.from_list([{"text":text,"label":label2int_dct["mc"][label]} for text, label in zip(mc_test_sents, mc_test_labels)]),
            "holdout": Dataset.from_list([{"text":text,"label":label2int_dct["mc"][label]} for text, label in zip(mc_ho_sents, mc_ho_labels)])
        })
        ############
        if save:
            bn_path = output_dir+f"/{e}_bn"
            mc_path = output_dir+f"/{e}_mc"
            if not os.path.exists(bn_path):
                os.makedirs(bn_path)
            if not os.path.exists(mc_path):
                os.makedirs(mc_path)
            bn_ds.save_to_disk(bn_path)
            mc_ds.save_to_disk(mc_path)
    return bn_ds, mc_ds

def main(sentences, labels, r=9):
    models = {
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1":'bert', 
        #"dunzhang/stella_en_1.5B_v5":'stella', 
        #"Alibaba-NLP/gte-Qwen2-1.5B-instruct":'qwen', 
        #"Alibaba-NLP/gte-large-en-v1.5":'glarg', 
        #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":'minilm'
        }

        #############
    sims = [3,6,9]
    int2label_dct, label2int_dct = load_labelintdcts()
    bn_ds, mc_ds = create_dsdict(sentences, labels, label2int_dct, amt=sims, save=True)
    for e in sims:
        for model in models:
            torch.cuda.empty_cache()
            #try:
            finetune_automodel(bn_ds, int2label_dct["bn"], label2int_dct["bn"], "bn", model_name=model, dev='cuda', rstate=e)
            print(f"\nCompleted {model} binary model.")
            #except Exception as e:
            #    print(f"\n{model} binary model failed due to {e}")
            torch.cuda.empty_cache()
            #try:
            finetune_automodel(mc_ds, int2label_dct["mc"], label2int_dct["mc"], "mc", model_name=model, dev='cuda', rstate=e)
            print(f"\nCompleted {model} multiclass model.")
            #except Exception as e:
            #    print(f"\n{model} multiclass model failed due to {e}")
    print('all done')

if __name__ == "__main__":
    with open(input_dir+"/19Jan25_firstdatarev.json","r", encoding="utf-8") as f:
        dcno_json = json.load(f)
    with open(input_dir+"/27Jan25_query_checked.json","r", encoding="utf-8") as f:
        qry_json = json.load(f)
    sents1, labels1 = dcno_to_sentlab(dcno_json)
    sents2, labels2 = dcno_to_sentlab(qry_json)
    # merge original and augmented datasets
    sents2.extend(sents1)
    labels2.extend(labels1)
    all_sents, all_labs = remove_duplicates(group_duplicates(sents2,labels2,thresh=90))
    #
    main(all_sents, all_labs, r=9)


