from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, cohen_kappa_score
from sentence_transformers import SentencesDataset, SentenceTransformer, InputExample
from sentence_transformer import EarlyStoppingSentenceTransformer
import torch
from torch.utils.data import DataLoader
from torch import device
import torch.nn as nn
from sentence_transformers import losses
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import os
import json
import time
import random
from rapidfuzz import fuzz
from latent_embeddings_classifier import encode_all_sents
import numpy as np
import math
from transformers import AutoModel, BitsAndBytesConfig, AutoTokenizer, AutoModelForSequenceClassification
from optimum.gptq import GPTQQuantizer, load_quantized_model

cwd = os.getcwd()
output_dir =  cwd+"/outputs/models"
input_dir =  cwd+"/inputs"

from run_classifiers import group_duplicates, remove_duplicates, dcno_to_sentlab, gen_bn_sentlab, gen_mc_sentlab, classify_svm
from loops import SoftmaxClassifier
from custom_evaluator import CustomLabelAccuracyEvaluator

# from loops.py
def build_data_samples(X_train, label2int, y_train):
    train_samples = []
    for sent, label in zip(X_train, y_train):
        label_id = label2int[label]
        train_samples.append(InputExample(texts=[sent], label=label_id))
    return train_samples

def fbqvdo_finetune(sentences, labels, mode, model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1", train_batch_size=16, dev='cuda', rstate=9, oom=False):
    epochs = 10
    print(f'\nLoading model {model_name}\n')
    try:
        model = SentenceTransformer(model_name, device=dev) 
        #model = EarlyStoppingSentenceTransformer(model_name, device=dev)
    except:
        model = SentenceTransformer(model_name, device=dev, trust_remote_code=True) 
        #model = EarlyStoppingSentenceTransformer(model_name, device=dev, trust_remote_code=True)
    print('Model loaded')
    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels, stratify=labels, test_size=0.2, random_state=rstate)
    label2int = dict(zip(set(labels), range(len(set(labels)))))
    #
    train_samples = build_data_samples(train_sents, label2int, train_labels)
    dev_samples = build_data_samples(test_sents, label2int, test_labels)
    #
    if oom:
        model.max_seq_length = 128
        model.gradient_checkpointing_enable()
        train_batch_size=8
    # Train set config
    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=train_batch_size)
    # Dev set config
    dev_dataset = SentencesDataset(dev_samples, model=model)
    dev_dataloader = DataLoader(
        dev_dataset, shuffle=True, batch_size=train_batch_size)
    # Define the way the loss is computed
    classifier = SoftmaxClassifier(model=model,
                                   sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                   num_labels=len(label2int)).to(dev)
    warmup_steps = math.ceil(len(train_dataset) * 10 / train_batch_size * 0.1)
    # Train the model
    start = time.time()
    dev_evaluator = CustomLabelAccuracyEvaluator(dataloader=dev_dataloader, softmax_model=classifier,
                                                 name='lae-dev', label_names=set(labels))
    print('Begin model fitting')
    model.fit(train_objectives=[(train_dataloader, classifier)],
              evaluator=dev_evaluator,
              epochs=epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              #output_path=f"/./{model.split('/')[0]}_epochs_2_rstate_{rstate}",
              optimizer_params=
                {
                    'lr': 2e-5, 
                    #'correct_bias': True
                },
              #baseline=0.001,
              #patience=5,
              )
    model.save(output_dir+f"/{model_name.split('/')[-1]}_{mode}_epochs_{epochs}_rstate_{rstate}.pt")
    end = time.time()
    print(f'\nDone in {(end-start)/60} min')
'''
def quantize_finetune(sentences, labels, mode, model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1", train_batch_size=16, dev = 'cuda', rstate=9, oom=False):
    print(f'Loading model {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16)
    # block_name_to_quantize = "model.decoder.layers", model_seqlen = 2048
    quantizer = GPTQQuantizer(bits=8, dataset="c4")
    quantized_model = quantizer.quantize_model(model, tokenizer)
    quantizer.save(quantized_model,output_dir+f'/{model_name}_quant')
    #
    try:
        model = SentenceTransformer(model_name, device=dev) 
    except:
        model = SentenceTransformer(model_name, device=dev, trust_remote_code=True) 
    print('Model loaded')
    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels, stratify=labels, test_size=0.2, random_state=rstate)
    label2int = dict(zip(set(labels), range(len(set(labels)))))
    #
    train_samples = build_data_samples(train_sents, label2int, train_labels)
    dev_samples = build_data_samples(test_sents, label2int, test_labels)
    # Train set config
    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=train_batch_size)
    # Dev set config
    dev_dataset = SentencesDataset(dev_samples, model=model)
    dev_dataloader = DataLoader(
        dev_dataset, shuffle=True, batch_size=train_batch_size)
    # Define the way the loss is computed
    classifier = SoftmaxClassifier(model=model,
                                   sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                   num_labels=len(label2int))
    warmup_steps = math.ceil(len(train_dataset) * 10 / train_batch_size * 0.1)
    # Train the model
    start = time.time()
    dev_evaluator = CustomLabelAccuracyEvaluator(dataloader=dev_dataloader, softmax_model=classifier,
                                                 name='lae-dev', label_names=set(labels))
    print('Begin model fitting')
    if oom:
        model.max_seq_length = 128
        model.gradient_checkpointing_enable()
    model.fit(train_objectives=[(train_dataloader, classifier)],
              evaluator=dev_evaluator,
              epochs=10,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              #output_path=f"/./{model.split('/')[0]}_epochs_2_rstate_{rstate}",
              optimizer_params=
                {
                    'lr': 2e-5, 
                    #'correct_bias': True
                },
              #baseline=0.001,
              #patience=5,
              )
    model.save(output_dir+f"/{model_name.split('/')[-1]}_{mode}_epochs_10_rstate_{rstate}.pt")
    end = time.time()
    print(f'Done in {(end-start)/60} min')

def custom_collate(batch):
    sentences = [item.texts[0] for item in batch]
    #sentences = [item.texts for item in batch]
    labels = torch.tensor([item.label for item in batch], dtype=torch.float32)
    return sentences, labels

def oom_finetune(sentences, labels, mode, model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1", train_batch_size=16, dev = 'cuda', rstate=9):
    epochs = 10
    print(f'Loading model {model_name}')
    # accounting for massive model size
    # configure quantization
    bb_config = BitsAndBytesConfig(
        # 8-bit
        #load_in_8bit=True,
        #llm_int8_threshold=6.0, 
        #llm_int8_enable_fp32_cpu_offload=True,
        # 4-bit
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  
    )
    # create quantized version of model
    quantized_model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config = bb_config,
        torch_dtype=torch.float16
    )
    # load pretrained version of model
    try:
        model = SentenceTransformer(model_name, device=dev)
    except:
        model = SentenceTransformer(model_name, device=dev, trust_remote_code=True)
    # Replace with quantized model
    model._first_module().auto_model = quantized_model
    #print(quantized_model.hf_device_map)
    print('Model loaded')
    # train test split with specific random state which will be used to evaluate model later
    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels, stratify=labels, test_size=0.2, random_state=rstate)
    label2int = dict(zip(set(labels), range(len(set(labels)))))
    # convert sentences 
    train_samples = build_data_samples(train_sents, label2int, train_labels)
    dev_samples = build_data_samples(test_sents, label2int, test_labels)
    # Train set config
    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_batch_size,
        collate_fn=custom_collate  
    )
    # Dev set config
    dev_dataset = SentencesDataset(dev_samples, model=model)
    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=True,
        batch_size=train_batch_size,
        collate_fn=custom_collate
    )
    # Define the way the loss is computed
    classifier = SoftmaxClassifier(model=model,
                                   sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                   num_labels=len(label2int)
                                   ).to(torch.float16)
    dev_evaluator = CustomLabelAccuracyEvaluator(dataloader=dev_dataloader, softmax_model=classifier,
                                                 name='lae-dev', label_names=set(labels))
    warmup_steps = math.ceil(len(train_dataset) * 10 / train_batch_size * 0.1)  # 10% of train data for warm-up
    # Train the model
    start = time.time()
    # moved away from model.fit() due to quantization/ mixed precision/ gradient scaling
    model.train()
    classifier.train()
    # fixing "only tensors of floating point and complex dtype can require gradients"
    # while maintaining torch.float16 dtype for the most part bc model is so big
    # but gradient computations and loss need float32
    for param in model.parameters():
        if param.is_floating_point():
            if param.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                try:
                    param.data = param.data.to(torch.float16)
                except Exception as e:
                    print(f"Skipping conversion for {param.shape}, dtype={param.dtype} due to: {e}")
            param.requires_grad = True
    for param in classifier.parameters():
        if param.is_floating_point():
            if param.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                try:
                    param.data = param.data.to(torch.float16)
                except Exception as e:
                    print(f"Skipping conversion for {param.shape}, dtype={param.dtype} due to: {e}")
            param.requires_grad = True
    # scaling gradient [for mixed precision]
    scaler = torch.cuda.amp.GradScaler()
    # custom loaded optimier
    optimizer = torch.optim.AdamW(list(set(list(model.parameters()) + list(classifier.parameters()))), lr=2e-5)
    # scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    # externally include baseline and patience standards 
    best_score = -float('inf')
    no_improve_epochs = 0
    baseline = 0.001
    patience = 5
    # externally initialize steps
    step = 0
    evaluation_steps = 1000
    #
    print("Begin model fitting")
    for epoch in range(epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                b_sents, b_labs = batch
                print(0)
                tok_sents = model.tokenizer(b_sents, padding=True, truncation=True, return_tensors="pt").to(model.device)
                print(1)
                outputs = model(tok_sents)["sentence_embedding"]
                print(2)
                print(b_labs)
                # classifier 
                #loss = classifier(outputs.to(torch.float32), b_labs)
                #loss = classifier(outputs, b_labs)
                loss_fxn = nn.BCEWithLogitsLoss()
                loss = loss_fxn(outputs.squeeze(1), b_labs)
                print(3)
            # avoid underflow
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step+=1
            # triggers periodical evaluation
            if step % evaluation_steps == 0:
                # save dev evaluator score to evaluate for further training
                score = dev_evaluator(model, classifier)
                if score > best_score + baseline:
                    best_score = score
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in {no_improve_epochs} epochs. Patience is {round(no_improve_epochs/patience)*100}% gone.")
                if no_improve_epochs >= patience:
                    print("We were patient. That's enough of that.")
                    break 
        if no_improve_epochs >= patience:  
            print("We were patient. That's enough of that.")
            break 
    model.save(output_dir+f"/{model_name.split('/')[-1]}_{mode}_epochs_{epochs}_rstate_{rstate}.pt")
    end = time.time()
    print(f'Done in {(end-start)/60} min')

def oom_stella(sentences, labels, mode, model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1", train_batch_size=16, dev = 'cuda', rstate=9):
    epochs = 10
    print(f'Loading model {model_name}')
    # accounting for massive model size
    # configure quantization
    bb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  
    )
    # create quantized version of model
    quantized_model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config = bb_config,
        torch_dtype=torch.float16
    )
    # load pretrained version of model
    try:
        model = SentenceTransformer(model_name, device=dev)
    except:
        model = SentenceTransformer(model_name, device=dev, trust_remote_code=True)
    # Replace with quantized model
    model._first_module().auto_model = quantized_model
    #print(quantized_model.hf_device_map)
    print('Model loaded')
    # train test split with specific random state which will be used to evaluate model later
    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,labels, stratify=labels, test_size=0.2, random_state=rstate)
    label2int = dict(zip(set(labels), range(len(set(labels)))))
    # convert sentences 
    train_samples = build_data_samples(train_sents, label2int, train_labels)
    dev_samples = build_data_samples(test_sents, label2int, test_labels)
    # Train set config
    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_batch_size,
        collate_fn=custom_collate  
    )
    # Dev set config
    dev_dataset = SentencesDataset(dev_samples, model=model)
    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=True,
        batch_size=train_batch_size,
        collate_fn=custom_collate
    )
    # Define the way the loss is computed
    classifier = SoftmaxClassifier(model=model,
                                   sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                   num_labels=len(label2int)
                                   ).to(torch.float16)
    dev_evaluator = CustomLabelAccuracyEvaluator(dataloader=dev_dataloader, softmax_model=classifier,
                                                 name='lae-dev', label_names=set(labels))
    warmup_steps = math.ceil(len(train_dataset) * 10 / train_batch_size * 0.1)  # 10% of train data for warm-up
    # Train the model
    start = time.time()
    # moved away from model.fit() due to quantization/ mixed precision/ gradient scaling
    model.train()
    classifier.train()
    # fixing "only tensors of floating point and complex dtype can require gradients"
    # while maintaining torch.float16 dtype for the most part bc model is so big
    # but gradient computations and loss need float32
    for param in model.parameters():
        if param.is_floating_point():
            if param.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                try:
                    param.data = param.data.to(torch.float16)
                except Exception as e:
                    print(f"Skipping conversion for {param.shape}, dtype={param.dtype} due to: {e}")
            param.requires_grad = True
    for param in classifier.parameters():
        if param.is_floating_point():
            if param.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                try:
                    param.data = param.data.to(torch.float16)
                except Exception as e:
                    print(f"Skipping conversion for {param.shape}, dtype={param.dtype} due to: {e}")
            param.requires_grad = True
    # scaling gradient [for mixed precision]
    scaler = torch.cuda.amp.GradScaler()
    # custom loaded optimier
    optimizer = torch.optim.AdamW(list(set(list(model.parameters()) + list(classifier.parameters()))), lr=2e-5)
    # scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    # externally include baseline and patience standards 
    best_score = -float('inf')
    no_improve_epochs = 0
    baseline = 0.001
    patience = 5
    # externally initialize steps
    step = 0
    evaluation_steps = 1000
    #
    print("Begin model fitting")
    for epoch in range(epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                b_sents, b_labs = batch
                print(0)
                tok_sents = model.tokenizer(b_sents, padding=True, truncation=True, return_tensors="pt").to(model.device)
                print(1)
                outputs = model(tok_sents)["sentence_embedding"]
                print(2)
                print(b_labs)
                # classifier 
                #loss = classifier(outputs.to(torch.float32), b_labs)
                #loss = classifier(outputs, b_labs)
                loss_fxn = nn.BCEWithLogitsLoss()
                loss = loss_fxn(outputs.squeeze(1), b_labs)
                print(3)
            # avoid underflow
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step+=1
            # triggers periodical evaluation
            if step % evaluation_steps == 0:
                # save dev evaluator score to evaluate for further training
                score = dev_evaluator(model, classifier)
                if score > best_score + baseline:
                    best_score = score
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in {no_improve_epochs} epochs. Patience is {round(no_improve_epochs/patience)*100}% gone.")
                if no_improve_epochs >= patience:
                    print("We were patient. That's enough of that.")
                    break 
        if no_improve_epochs >= patience:  
            print("We were patient. That's enough of that.")
            break 
    model.save(output_dir+f"/{model_name.split('/')[-1]}_{mode}_epochs_{epochs}_rstate_{rstate}.pt")
    end = time.time()
    print(f'Done in {(end-start)/60} min')
'''
def main(sentences, labels, r=69):
    models = {
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1":'bert', 
        #"dunzhang/stella_en_1.5B_v5":'stella', 
        #"Alibaba-NLP/gte-Qwen2-1.5B-instruct":'qwen', 
        #"Alibaba-NLP/gte-large-en-v1.5":'glarg', 
        #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":'minilm'
        }
    bn_sents, bn_labels = gen_bn_sentlab(sentences, labels, sanity_check=False)
    mc_sents, mc_labels = gen_mc_sentlab(sentences, labels, sanity_check=False)

    bn_ft_sents, bn_ho_sents, bn_ft_labels, bn_ho_labels = train_test_split(bn_sents, bn_labels, stratify=bn_labels, test_size=0.2, random_state=r)
    mc_ft_sents, mc_ho_sents, mc_ft_labels, mc_ho_labels = train_test_split(mc_sents, mc_labels, stratify=mc_labels, test_size=0.25, random_state=r)
    '''
    for model in models:
        for rstate in range(10):
            torch.cuda.empty_cache()
            fbqvdo_finetune(bn_ft_sents, bn_ft_labels, "bn", model_name=model, dev='cuda', rstate=rstate)
    '''
    for model in models:
        for rstate in range(10):
            torch.cuda.empty_cache()
            fbqvdo_finetune(mc_ft_sents, mc_ft_labels, "mc", model_name=model, dev='cuda', rstate=rstate)
    '''
    for model in models:
        #try:
        torch.cuda.empty_cache()
        #oom_finetune(bn_sents, bn_labels,"bn", model_name=model, dev = 'cuda', rstate=0)
        quantize_finetune(bn_sents, bn_labels,"bn", model_name=model, dev = 'cuda', rstate=0)
        #except Exception as e:
        #    print(f'Error in {model}: \n{e} \nTrying with quantization and lower batch size.')
        #    try:
        #        torch.cuda.empty_cache()
        #        oom_finetune(bn_sents, bn_labels,"bn", model_name=model, train_batch_size=2, dev = 'cuda', rstate=0)
        #    except Exception as e:
        #        print(f'Error in {model}: \n{e} \nUnsuccessful run.')
    for model in models:
        try:
            torch.cuda.empty_cache()
            oom_finetune(mc_sents, mc_labels,"mc", model_name=model, dev = 'cuda', rstate=0)
        except Exception as e:
            print(f'Error in {model}: \n{e} \nTrying with quantization and lower batch size.')
            try:
                torch.cuda.empty_cache()
                oom_finetune(mc_sents, mc_labels,"mc", model_name=model, train_batch_size=2, dev = 'cuda', rstate=0)
            except Exception as e:
                print(f'Error in {model}: \n{e} \nUnsuccessful run.')
    '''
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
    # 0 for bn
    # 9 for mc
    main(all_sents, all_labs, r=69)


