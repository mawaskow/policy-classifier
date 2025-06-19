from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, AdamW, get_linear_schedule_with_warmup
import evaluate
from sklearn.metrics import roc_curve
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
import numpy as np
import torch.nn as nn
#from classifier.finetune import finetune_roberta, load_labelintdcts, create_dsdict, create_om_dsdict
#from classifier.run_classifiers import group_duplicates, remove_duplicates, dcno_to_sentlab
from finetune import load_labelintdcts, create_dsdict, create_om_dsdict
from run_classifiers import group_duplicates, remove_duplicates, dcno_to_sentlab
import gc
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import wandb
import os, json
cwd = os.getcwd() # should be base directory of repository
import time
import torch
from datasets import DatasetDict, Dataset
import multiprocessing as mp
import subprocess

def plot_roc(labels, probs, mode, rstate, output_dir, int2label):
    if mode=="bn":
        fpr, tpr, thresholds = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        #
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='teal')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve {mode} {rstate}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/ROC_{mode}_{rstate}.png")
        #plt.show()
    elif mode=="mc":
        n_classes = probs.shape[1]
        y_test_bin = label_binarize(labels, classes=range(n_classes))
        # Compute ROC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Plot all
        plt.figure()
        colors = ['firebrick', 'darkorange', 'gold', 'yellowgreen', "deepskyblue", "slateblue"]
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color=colors[i], label=f"Class {int2label[i]} (AUC = {roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve {mode} {rstate}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/ROC_{mode}_{rstate}.png")
        #plt.show()
    else:
        n_classes = probs.shape[1]
        y_test_bin = label_binarize(labels, classes=range(n_classes))
        # Compute ROC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Plot all
        plt.figure()
        colors = ['firebrick', 'darkorange', 'gold', 'yellowgreen', "deepskyblue", "slateblue", "mediumorchid"]
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color=colors[i], label=f"Class {int2label[i]} (AUC = {roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve {mode} {rstate}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/ROC_{mode}_{rstate}.png")
        #plt.show()

class WeightedTrainer(Trainer):
    def __init__(self, *args, loss_ratio=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_ratio = loss_ratio
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):#):#
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Define class weights and loss
        weights = torch.tensor(self.loss_ratio).to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits, labels)
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def finetune_roberta(datasetdct, int2label, label2int, mode, model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1", dev='cuda', output_dir=f"{os.getcwd()}/outputs/models", hyperparams=False, report_to="none", span=False):
    '''
    '''
    if not hyperparams:
        hyperparams = {
            "epochs":10, 
            "r":9,
            "lr":2e-5,
            "batch_size":16,
            "loss":False,
            "oversampling":False
            }
    epochs = hyperparams["epochs"]
    rstate = hyperparams["r"]
    lr = hyperparams["lr"]
    batch_size = hyperparams["batch_size"]
    loss_ratio = hyperparams["loss"]
    ovs_ratio = hyperparams["oversampling"]
    start = time.time()
    num_lbs = len(list(int2label))
    print(f'\nLoading model {model_name}\n')
    print("Tokenizing")
    if not span:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif span == "bn":
        tokenizer = AutoTokenizer.from_pretrained("../../inputs/polianna_models/inputs/polianna_models/paraphrase-xlm-r-multilingual-v1_bn_e2_r9.pt")
    else:
        tokenizer = AutoTokenizer.from_pretrained("../../inputs/polianna_models/inputs/polianna_models/paraphrase-xlm-r-multilingual-v1_mc_e2_r9.pt")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)#.to(dev)
    tokenized_test = datasetdct["test"].map(preprocess_function, batched=True)
    if not ovs_ratio:
        tokenized_train = datasetdct["train"].map(preprocess_function, batched=True)
    else:
        train_sents = datasetdct["train"]["text"]
        train_labels = datasetdct["train"]["label"]
        ros = RandomOverSampler(sampling_strategy='auto', random_state=rstate)
        train_texts_resampled, train_labels_resampled = ros.fit_resample(np.array(train_sents).reshape(-1, 1), np.array(train_labels))
        train_texts_resampled, train_labels_resampled = shuffle(train_texts_resampled, train_labels_resampled, random_state=rstate)
        flattened_texts = list(train_texts_resampled.flatten())
        conv_dct = {"text":flattened_texts, "label":train_labels_resampled}
        conv_ds = Dataset.from_dict(conv_dct)
        tokenized_train = conv_ds.map(preprocess_function, batched=True)
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    recall = evaluate.load("recall")
    metric_log = []
    def calc_metrics(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=1)
        metrics = {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            #"f1": f1.compute(predictions=predictions, references=labels, average="weighted" if mode=="mc" else "binary")["f1"],
            "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
            "recall": recall.compute(predictions=predictions, references=labels, average="weighted")["recall"]
        }
        metric_log.append(metrics)
        return metrics
    print("Loading model")
    #
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_lbs,id2label=int2label, label2id=label2int).to(dev)
    except:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_lbs,id2label=int2label, label2id=label2int, trust_remote_code=True).to(dev)
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=9,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="best",
        logging_strategy="epoch",
        optim="adamw_torch",
        load_best_model_at_end=True,
        report_to= report_to,
        run_name=f"{output_dir.split('_')[-2]}{rstate}" if report_to == "wandb" else "X"
    )
    #
    if loss_ratio:
        trainer = WeightedTrainer(
            loss_ratio=loss_ratio,
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            processing_class=tokenizer,
            compute_metrics=calc_metrics,
            #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
        )
    else:
        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
                processing_class=tokenizer,
                compute_metrics=calc_metrics,
            )
    print("Training")
    trainer.train()
    train_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    eval_losses = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
    print("Saving")
    model_fn = f"{model_name.split('/')[-1]}_{mode}_e{epochs}_r{rstate}.pt"
    trainer.save_model(output_dir+f"/{model_fn}")
    #
    tokenized_ho = datasetdct["holdout"].map(preprocess_function, batched=True)
    predictions = trainer.predict(tokenized_ho)
    logits = predictions.predictions
    labels = predictions.label_ids
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    #
    plot_roc(labels, probs, mode, rstate, output_dir, int2label)
    metric_log.append({"train_loss":train_losses})
    metric_log.append({"eval_loss": eval_losses})
    metrics = calc_metrics((logits, labels))
    with open(output_dir+f"/{model_fn}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metric_log, f, ensure_ascii=False, indent=4)
    end = time.time()
    print(metrics)
    print(f"\nSaved {model_name.split('/')[-1]}_{mode}_e{epochs}_r{rstate}.")
    print(f'\nDone in {round((end-start)/60,2)} min')
    print(rstate, "D", torch.cuda.memory_allocated())
    del model
    del trainer
    del tokenizer
    del predictions
    del logits
    del labels
    torch.cuda.empty_cache()
    gc.collect()
    print(rstate, "E", torch.cuda.memory_allocated())
    return metrics

def gernerate_dsdicts(input_dir, mode="split"):
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
    int2label_dct, label2int_dct = load_labelintdcts()
    sims = range(10)
    if mode=="split":
        create_dsdict(all_sents, all_labs, label2int_dct, amt=sims, save=True, output_dir=input_dir)
    else:
        create_om_dsdict(all_sents, all_labs, label2int_dct, amt=sims, save=True, output_dir=input_dir)

def main():
    cwd = os.getcwd()
    output_dir = cwd+"/outputs/"
    input_dir = cwd+"/inputs/"
    int2label_dct, label2int_dct = load_labelintdcts()
    metriclog = {}
    exps = range(10)#[0,3,6]#[6]#[0,3,6,9]#
    torch.cuda.empty_cache()
    gc.collect()
    '''
    letter = "N"
    mode = "bn"
    for r in exps:
        print(f"\n\nBeginning run {letter} {mode} {r}\n")
        output_dir = cwd+f"/outputs/fting_{letter}_{mode}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ds = DatasetDict.load_from_disk(f"{input_dir}/ds_{r}_{mode}")
        hyper = {
            "epochs":5, 
            "r":r, 
            "lr":2E-5,
            "batch_size":16,
            "loss":None,
            "oversampling":None,
            "span_step":None
            }
        print(r, "A", torch.cuda.memory_allocated())
        torch.cuda.empty_cache()
        gc.collect()
        print(r, "B", torch.cuda.memory_allocated())
        #wandbrun = wandb.init(config=hyper, group=letter, name=f"{letter}{r}", reinit='create_new')
        metrics = finetune_roberta(ds, int2label_dct[mode], label2int_dct[mode], mode, model_name='Alibaba-NLP/gte-base-en-v1.5', dev='cuda', output_dir=output_dir, hyperparams=hyper)
        print(r, "C", torch.cuda.memory_allocated())
        #wandbrun.finish()
        metriclog[f'{mode}_{r}'] = metrics
        hyp_rpt = {mode:hyper}
        with open(output_dir+f"/run_details.json", "w", encoding="utf-8") as f:
            json.dump(hyp_rpt, f, ensure_ascii=False, indent=4)
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3)
        print(r, "D", torch.cuda.memory_allocated())
        print(f"\n\nCompleted run {letter} {mode} {r}\n")
    '''    
    '''
    letter = "O"
    mode = "mc"
    for r in exps:
        output_dir = cwd+f"/outputs/fting_{letter}_{mode}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ds = DatasetDict.load_from_disk(f"{input_dir}/ds_{r}_{mode}")
        hyper = {
            "epochs":15, 
            "r":r, 
            "lr":2E-5,
            "batch_size":16,
            "loss":None,
            "oversampling":None,
            "span_step":None
            }
        torch.cuda.empty_cache()
        gc.collect()
        #wandbrun = wandb.init(config=hyper, group=letter, name=f"{letter}{r}", reinit='create_new')
        metrics = finetune_roberta(ds, int2label_dct[mode], label2int_dct[mode], mode, model_name='Alibaba-NLP/gte-base-en-v1.5', dev='cuda', output_dir=output_dir, hyperparams=hyper)
        #wandbrun.finish()
        metriclog[f'{mode}_{r}'] = metrics
        hyp_rpt = {mode:hyper}
        with open(output_dir+f"/run_details.json", "w", encoding="utf-8") as f:
            json.dump(hyp_rpt, f, ensure_ascii=False, indent=4)
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3)
        print(r, "D", torch.cuda.memory_allocated())
    print(metriclog)
    '''

if __name__ == '__main__':
    #letter = "R"
    #mode = "bn"
    exps = range(10)#[0,3,6]#[6]#[0,3,6,9]#
    #model_name = "xlm-roberta-base"
    #model_name = "microsoft/Multilingual-MiniLM-L12-H384"
    #model_name = "distilbert-base-multilingual-cased"
    #model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    #model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
    #
    cwd = os.getcwd()
    #output_dir = cwd+"/outputs/"
    output_dir = "E:/PhD/2June2025/"
    input_dir = cwd+"/inputs/"
    torch.cuda.empty_cache()
    gc.collect()
    int2label_dct, label2int_dct = load_labelintdcts()
    '''
    hyper = {"bn":{
                "epochs":5, 
                "lr":2E-5,
                "batch_size":16,
                "loss":None,
                "oversampling":None,
                "span_step":None,
                "int2label":int2label_dct["bn"],
                "label2int":label2int_dct["bn"]
            },
             "mc":{
                "epochs":15, 
                "lr":2E-5,
                "batch_size":16,
                "loss":None,
                "oversampling":None,
                "span_step":None,
                "int2label":int2label_dct["mc"],
                "label2int":label2int_dct["mc"]
            }
    }
    '''
    '''
    letters = ["G", "H", "I", "J", "K", "L"]
    models =["sentence-transformers/paraphrase-xlm-r-multilingual-v1", "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/paraphrase-mpnet-base-v2", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "thenlper/gte-base", "intfloat/e5-base-v2"]
    for mode in ["bn", "mc"]:
        for i in range(len(models)):
            letter = letters[i]
            model_name = models[i]
            for r in exps:
                while not torch.cuda.is_available():
                    print("Cuda unavailable")
                    time.sleep(3)
                print("\nCuda freed!")
                st = time.time()
                print(f"\n--- Starting run {model_name} {letter} {r} ---")
                print("Start", torch.cuda.memory_allocated())
                modeldir = os.path.join(output_dir, f"fting_{letter}_{mode}")
                os.makedirs(modeldir, exist_ok=True)

                subprocess.run([
                    "python", "classifier/oneft.py",
                    str(r),
                    mode,
                    letter,
                    input_dir,
                    modeldir,
                    model_name,
                    "loss"
                ],
                    check=True, capture_output=True, text=True)
                print(f"\n--- Finished run {model_name} {letter} {r} ---")
                print(f'\nDone in {round((time.time()-st)/60,2)} min')
                print("End", torch.cuda.memory_allocated())
                time.sleep(2)
    '''
    ############
    letters = ["P", "Q", "R"]
    models =["sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "thenlper/gte-base", "intfloat/e5-base-v2"]
    for mode in ["mc"]:
        for i in range(len(models)):
            letter = letters[i]
            model_name = models[i]
            for r in exps:
                while not torch.cuda.is_available():
                    print("Cuda unavailable")
                    time.sleep(3)
                print("\nCuda freed!")
                st = time.time()
                print(f"\n--- Starting run {model_name} {letter} {r} ---")
                print("Start", torch.cuda.memory_allocated())
                modeldir = os.path.join(output_dir, f"fting_{letter}_{mode}")
                os.makedirs(modeldir, exist_ok=True)

                subprocess.run([
                    "python", "classifier/oneft.py",
                    str(r),
                    mode,
                    letter,
                    input_dir,
                    modeldir,
                    model_name,
                    "os"
                ],
                    check=True, capture_output=True, text=True)
                print(f"\n--- Finished run {model_name} {letter} {r} ---")
                print(f'\nDone in {round((time.time()-st)/60,2)} min')
                print("End", torch.cuda.memory_allocated())
                time.sleep(2)



    