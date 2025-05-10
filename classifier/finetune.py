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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, AdamW, get_linear_schedule_with_warmup
import evaluate
from datasets import Dataset, DatasetDict
from collections import Counter
from peft import LoraConfig, get_peft_model
from sklearn.utils import shuffle

cwd = os.getcwd()
output_dir = cwd+"/outputs/automodels_nofreeze"
input_dir = cwd+"/inputs"

#from run_classifiers import group_duplicates, remove_duplicates, dcno_to_sentlab, gen_bn_sentlab, gen_mc_sentlab
from classifier.run_classifiers import group_duplicates, remove_duplicates, dcno_to_sentlab, gen_bn_sentlab, gen_mc_sentlab

def finetune_roberta(datasetdct, int2label, label2int, mode, model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1", dev='cuda', output_dir=f"{os.getcwd()}/outputs/models", hyperparams=False, oversampling=False):
    '''
    '''
    if not hyperparams:
        hyperparams = {
            "epochs":10, 
            "r":9,
            "lr":2e-5,
            "batch_size":16
            }
    epochs = hyperparams["epochs"]
    rstate = hyperparams["r"]
    lr = hyperparams["lr"]
    batch_size = hyperparams["batch_size"]
    start = time.time()
    num_lbs = len(list(int2label))
    print(f'\nLoading model {model_name}\n')
    print("Tokenizing")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True).to(dev)
    tokenized_ds = datasetdct.map(preprocess_function, batched=True)
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    # recall = evaluate.load("recall")
    metric_log = []
    def calc_metrics(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=1)
        metrics = {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": f1.compute(predictions=predictions, references=labels, average="weighted" if mode=="mc" else "binary")["f1"],
            # "recall"
        }
        metric_log.append(metrics)
        return metrics
    print("Loading model")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_lbs,id2label=int2label, label2id=label2int).to(dev)
    if oversampling:
        # sampling_strategy='minority'?
        # or fine-tune the RandomOverSampler?
        # https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
        # shrinkage?
        train_sents = tokenized_ds["train"]["text"]
        train_labels = tokenized_ds["train"]["label"]
        ros = RandomOverSampler(sampling_strategy='auto', random_state=rstate)
        train_texts_resampled, train_labels_resampled = ros.fit_resample(np.array(train_sents).reshape(-1, 1), np.array(train_labels))
        train_texts_resampled, train_labels_resampled = shuffle(train_texts_resampled, train_labels_resampled, random_state=rstate)
        tokenized_resampled_ds = tokenizer(list(train_texts_resampled.flatten()), padding=True, truncation=True, return_tensors="pt")
        tokenized_resampled_ds["label"] = torch.tensor(train_labels_resampled)
        tokenized_ds["train"] = tokenized_resampled_ds
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=9,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="steps",
        save_strategy="best",
        optim="adamw_torch",
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        processing_class=tokenizer,
        compute_metrics=calc_metrics,
        #https://github.com/huggingface/transformers/issues/10845
        #compute_loss_func=nn.CrossEntropyLoss(),
        #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
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

def create_labelintdcts(bn_labels, mc_labels, save=False, output_dir=f"{os.getcwd()}/outputs/models"):
    '''
    Takes binary labels and multiclass labels lists and allows you to create and save the label-int conversion dictionaries
    '''
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
    '''
    Loads the int2label and label2int dictionaries, each containing the respctive dictionary under mode type "bn", "mc", or "om"
    '''
    int2label_dct = {
        "bn": {
            0: "non-incentive",
            1: "incentive"
        },
        "mc":{
            0: "Fine",
            1: "Supplies",
            2: "Technical_assistance",
            3: "Tax_deduction",
            4: "Credit",
            5: "Direct_payment"
        },
        "om":{
            0: "Non-Incentive",
            1: "Fine",
            2: "Supplies",
            3: "Technical_assistance",
            4: "Tax_deduction",
            5: "Credit",
            6: "Direct_payment"
        }
    }
    label2int_dct = {
        "bn": {
            "non-incentive": 0,
            "incentive": 1
        },
        "mc":{
            "Fine": 0,
            "Supplies": 1,
            "Technical_assistance": 2,
            "Tax_deduction": 3,
            "Credit": 4,
            "Direct_payment": 5
        },
        "om":{
            "Non-Incentive": 0,
            "Fine": 1,
            "Supplies": 2,
            "Technical_assistance": 3,
            "Tax_deduction": 4,
            "Credit": 5,
            "Direct_payment": 6
        }
    }
    return int2label_dct, label2int_dct

def create_dsdict(sentences, labels, label2int_dct, amt=range(10), save=False, output_dir=f"{os.getcwd()}/outputs/models"):
    '''
    Create and save DatasetDicts containing train, test, and holdout sets.
    60:20:20
    '''
    bn_sents, bn_labels = gen_bn_sentlab(sentences, labels, sanity_check=False)
    mc_sents, mc_labels = gen_mc_sentlab(sentences, labels, sanity_check=False)
    for e in amt:
        print(f"\nRound {e}\n")
        bn_ft_sents, bn_ho_sents, bn_ft_labels, bn_ho_labels = train_test_split(bn_sents, bn_labels, stratify=bn_labels, test_size=0.2, random_state=e)
        mc_ft_sents, mc_ho_sents, mc_ft_labels, mc_ho_labels = train_test_split(mc_sents, mc_labels, stratify=mc_labels, test_size=0.2, random_state=e)
        #
        bn_train_sents, bn_test_sents, bn_train_labels, bn_test_labels = train_test_split(bn_ft_sents, bn_ft_labels, stratify=bn_ft_labels, test_size=0.25, random_state=e)
        mc_train_sents, mc_test_sents, mc_train_labels, mc_test_labels = train_test_split(mc_ft_sents, mc_ft_labels, stratify=mc_ft_labels, test_size=0.25, random_state=e)
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
            bn_path = output_dir+f"/ds_{e}_bn"
            mc_path = output_dir+f"/ds_{e}_mc"
            if not os.path.exists(bn_path):
                os.makedirs(bn_path)
            if not os.path.exists(mc_path):
                os.makedirs(mc_path)
            bn_ds.save_to_disk(bn_path)
            print(f"Saved ds_{e}_bn")
            mc_ds.save_to_disk(mc_path)
            print(f"Saved ds_{e}_mc")
    return None

def create_om_dsdict(sentences, labels, label2int_dct, amt=range(10), save=False, output_dir=f"{os.getcwd()}/outputs/models"):
    '''
    Create and save DatasetDicts containing train, test, and holdout sets.
    60:20:20
    '''
    for e in amt:
        print(f"\nRound {e}\n")
        ft_sents, ho_sents, ft_labels, ho_labels = train_test_split(sentences, labels, stratify=labels, test_size=0.2, random_state=e)
        train_sents, test_sents, train_labels, test_labels = train_test_split(ft_sents, ft_labels, stratify=ft_labels, test_size=0.25, random_state=e)
        #
        ds = DatasetDict({
            "train": Dataset.from_list([{"text":text,"label":label2int_dct["om"][label]} for text, label in zip(train_sents, train_labels)]),
            "test": Dataset.from_list([{"text":text,"label":label2int_dct["om"][label]} for text, label in zip(test_sents, test_labels)]),
            "holdout": Dataset.from_list([{"text":text,"label":label2int_dct["om"][label]} for text, label in zip(ho_sents, ho_labels)])
        })
        ############
        if save:
            path = output_dir+f"/ds_{e}_om"
            if not os.path.exists(path):
                os.makedirs(path)
            ds.save_to_disk(path)
            print(f"Saved ds_{e}_om")
    return None

def main():
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
    for e in sims:
        bn_ds = DatasetDict.load_from_disk(output_dir+f"/ds_{e}_bn")
        mc_ds = DatasetDict.load_from_disk(output_dir+f"/ds_{e}_mc")
        for model in models:
            torch.cuda.empty_cache()
            #try:
            finetune_roberta(bn_ds, int2label_dct["bn"], label2int_dct["bn"], "bn", model_name=model, dev='cuda', rstate=e)
            print(f"\nCompleted {model} binary model.")
            #except Exception as e:
            #    print(f"\n{model} binary model failed due to {e}")
            torch.cuda.empty_cache()
            #try:
            finetune_roberta(mc_ds, int2label_dct["mc"], label2int_dct["mc"], "mc", model_name=model, dev='cuda', rstate=e)
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
    int2label_dct, label2int_dct = load_labelintdcts()
    sims = [3,6,9]
    create_dsdict(all_sents, all_labs, label2int_dct, amt=sims, save=True, output_dir=output_dir)
    main()


