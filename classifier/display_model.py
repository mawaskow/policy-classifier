import logging.config
from mako.template import Template
import json
import os
import glob, regex
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import pandas as pd
from tqdm import tqdm
import torch
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gc
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")

CWD = os.getcwd()
INPUT_PTH = CWD+"/inputs/"

IDLBLLIB = { 
    "bn":
        {"id2label": {
            "0": "non-incentive",
            "1": "incentive"
        },
        "label2id": {
            "incentive": 1,
            "non-incentive": 0
        }},
    "mc":{
        "id2label": {
            "0": "Fine",
            "1": "Supplies",
            "2": "Technical_assistance",
            "3": "Tax_deduction",
            "4": "Credit",
            "5": "Direct_payment"
        },
        "label2id": {
            "Credit": 4,
            "Direct_payment": 5,
            "Fine": 0,
            "Supplies": 1,
            "Tax_deduction": 3,
            "Technical_assistance": 2
        }},
    "om":{
        "id2label": {
            "0": "Non-Incentive",
            "1": "Fine",
            "2": "Supplies",
            "3": "Technical_assistance",
            "4": "Tax_deduction",
            "5": "Credit",
            "6": "Direct_payment"
        },
        "label2id": {
            "Credit": 5,
            "Direct_payment": 6,
            "Fine": 1,
            "Non-Incentive": 0,
            "Supplies": 2,
            "Tax_deduction": 4,
            "Technical_assistance": 3
        }
    }
}

def encode_all_sents(all_sents, sbert_model):
    '''
    modified from previous repository's latent_embeddings_classifier.py
    '''
    stacked = np.vstack([sbert_model.encode(sent) for sent in tqdm(all_sents)])
    return [torch.from_numpy(element).reshape((1, element.shape[0])) for element in stacked]

class ModelReport:
    def __init__(self, model_dir, cls_mode="model", ovs=False):
        self.model_dir = model_dir
        self.cls_mode=cls_mode
        self.model_name = model_dir.split("/")[-1][:-3]
        self.model_name = self.model_name.split("\\")[-1]
        self.callname = self.model_name
        self.mode = self.callname.split("_")[1]
        self.metrics = self.load_metrics()
        self.config = self.load_config()
        self.ovs = False
        try:
            self.id2label = self.config["id2label"]
            self.label2id = self.config["label2id"]
        except:
            self.id2label = IDLBLLIB[self.mode]["id2label"]
            self.label2id = IDLBLLIB[self.mode]["label2id"]
            self.ovs = ovs
        self.real = None
        self.predicted = None
        self.r = self.callname.split("_")[-1][1:]
        self.cls_report = {}
        self.cm = []
        self.om_report = {}
        self.om_mc_lbs = []
        self.val_loss = [0]
    def load_metrics(self):
        metrics_path = self.model_dir+'/metrics.json'
        with open(metrics_path) as f:
            return json.load(f)
    def load_config(self):
        config_path = self.model_dir+'/config.json'
        with open(config_path) as f:
            return json.load(f)
    def calculate_randps(self, ds_addr, eval_batch=32):
        torch.cuda.empty_cache()
        dev = 'cuda' if torch.cuda.is_available() else None
        dsdct = DatasetDict.load_from_disk(ds_addr)
        sentences = dsdct["holdout"]["text"]
        labels = dsdct["holdout"]["label"]
        prds_lst = []
        if self.cls_mode == "model":
            num_lbs = len(set(labels))
            print("Loading tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            print("Loading model")
            try:
                model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, num_labels=num_lbs,id2label=self.id2label, label2id=self.label2id).to(dev)
            except:
                model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, num_labels=num_lbs,id2label=self.id2label, label2id=self.label2id, trust_remote_code=True).to(dev)
            model.eval()
            print("Running model")
            for i in tqdm(range(0, len(sentences),eval_batch)):
                bsents = sentences[i : i + eval_batch]
                test_embs = tokenizer(bsents, truncation=True, padding=True, return_tensors="pt").to(dev)
                logits = model(**test_embs).logits
                prds = torch.max(logits,1).indices
                prds_lst.extend(prds.tolist())
                del test_embs, logits
                torch.cuda.empty_cache()
                gc.collect()
            print("Freeing memory")
            del model
            torch.cuda.empty_cache()
            gc.collect()
        elif self.cls_mode in ["svm","rf"]:
            train_sents = dsdct["train"]["text"]+dsdct["test"]["text"]
            train_labels = dsdct["train"]["label"]+dsdct["test"]["label"]
            if self.ovs:
                ros = RandomOverSampler(sampling_strategy='auto', random_state=self.r)
                train_texts_resampled, train_labels_resampled = ros.fit_resample(np.array(train_sents).reshape(-1, 1), np.array(train_labels))
                train_texts_resampled, train_labels = shuffle(train_texts_resampled, train_labels_resampled, random_state=self.r)
                train_sents = list(train_texts_resampled.flatten())
            try:
                model = SentenceTransformer(self.model_dir, device=dev)
            except:
                model = SentenceTransformer(self.model_dir, device=dev, trust_remote_code=True)
            train_embs = encode_all_sents(train_sents, model)
            print("Encoding test sentences.")
            test_embs = encode_all_sents(sentences, model)
            if self.cls_mode == "svm":
                clf = svm.SVC(gamma=0.001, C=100., random_state=int(self.r))
                clf.fit(np.vstack(train_embs), train_labels)
                prds_lst = [int(clf.predict(sent_emb)[0]) for sent_emb in test_embs]
            else:
                clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=int(self.r))
                clf.fit(np.vstack(train_embs), train_labels)
                prds_lst = [int(clf.predict(sent_emb)[0]) for sent_emb in test_embs]
            del model
        torch.cuda.empty_cache()
        gc.collect()
        self.real = labels
        self.pred = prds_lst
        with open(self.model_dir+f"/randp_{self.cls_mode}{'_ovs' if self.ovs else ''}.json", "w", encoding="utf-8") as f:
            json.dump({"real":labels,"pred":prds_lst}, f, ensure_ascii=False, indent=4)
    def load_randps(self, real, pred):
        self.real = real
        self.pred = pred
    def calc_metrics(self):
        cls_report = classification_report(self.real, self.pred, output_dict=True)
        self.cls_report={"overall":{}, "labels":{}}
        self.cls_report["overall"]=cls_report["weighted avg"]
        self.cls_report["overall"]["accuracy"]= cls_report["accuracy"]
        for label in list(self.id2label):
            self.cls_report["labels"][label] = cls_report[label]
        self.cm = confusion_matrix(self.real, self.pred, normalize="true")
        self.cls_report["cm"] = self.cm
        if self.mode in ["mc","om"]:
            self.cls_report["custom_acc"] = {}
            real_ar = np.array(self.real)
            pred_ar = np.array(self.pred)
            mask = real_ar != 3 if self.mode=="mc" else real_ar != 4
            notd_real = real_ar[mask]
            notd_pred = pred_ar[mask]
            acc_no_td = accuracy_score(notd_real, notd_pred)
            self.cls_report["custom_acc"]["without_td"] = acc_no_td
            #
            td_mask = real_ar == 3 if self.mode=="mc" else real_ar == 4
            td_real = real_ar[td_mask]
            td_pred = pred_ar[td_mask]
            acc_td = accuracy_score(td_real, td_pred)
            self.cls_report["custom_acc"]["only_td"] = acc_td
            #print(self.cls_report["custom_acc"])
    def calc_metrics_om2bnmc(self):
        self.om_report = {"bn":{}, "mc":{}}
        real_ar = np.array(self.real)
        pred_ar = np.array(self.pred)
        #
        real_bn = [0 if i==0 else 1 for i in real_ar]
        pred_bn = [0 if i==0 else 1 for i in pred_ar]
        real_mc = []
        pred_mc = []
        for i in range(len(real_ar)):
            if real_ar[i]!=0:
                real_mc.append(real_ar[i])
                pred_mc.append(pred_ar[i])
        #bn
        #print(set(real_ar), set(pred_ar))
        self.om_report["bn"]={"overall":{}, "labels":{}}
        cls_report_bn = classification_report(real_bn, pred_bn, output_dict=True)
        self.om_report["bn"]["overall"]=cls_report_bn["weighted avg"]
        self.om_report["bn"]["overall"]["accuracy"]= cls_report_bn["accuracy"]
        for label in range(2):
            self.om_report["bn"]["labels"][label] = cls_report_bn[str(label)]
        self.om_report["bn"]["cm"] = confusion_matrix(real_bn, pred_bn, normalize="true")
        #mc
        self.om_report["mc"]={"overall":{}, "labels":{}}
        cls_report_mc = classification_report(real_mc, pred_mc, output_dict=True)
        self.om_report["mc"]["overall"]=cls_report_mc["weighted avg"]
        self.om_report["mc"]["overall"]["accuracy"]= cls_report_mc["accuracy"]
        for label in range(1,7):
            try:
                self.om_report["mc"]["labels"][label] = cls_report_mc[str(label)]
                self.om_mc_lbs.append(label)
            except Exception as e:
                print(f"Could not add data for label {label} due to {e}. Setting values to 0.")
                self.om_report["mc"]["labels"][label] = {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
        self.om_report["mc"]["cm"] = confusion_matrix(real_mc, pred_mc, normalize="true")
        # cust acc
        self.om_report["mc"]["custom_acc"] = {}
        real_mc_ar = np.array(real_mc)
        pred_mc_ar = np.array(pred_mc)
        mask = real_mc_ar != 4
        notd_real = real_mc_ar[mask]
        notd_pred = pred_mc_ar[mask]
        acc_no_td = accuracy_score(notd_real, notd_pred)
        self.om_report["mc"]["custom_acc"]["without_td"] = acc_no_td
        td_mask = real_mc_ar == 4
        td_real = real_mc_ar[td_mask]
        td_pred = pred_mc_ar[td_mask]
        acc_td = accuracy_score(td_real, td_pred)
        # handle nan
        if acc_td!=acc_td:
            acc_td = 0
        self.om_report["mc"]["custom_acc"]["only_td"] = acc_td
    def plot_cfmtx(self):
        fs = 4 if self.mode=="bn" else 6 if self.mode=="mc" else 7
        #cmap = plt.get_cmap("YlGn")
        import matplotlib.colors as colors
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap
        temp_cmap = plt.get_cmap("viridis_r")
        cmap = truncate_colormap(temp_cmap, 0.01, 0.55)
        if self.mode =="bn":
            labels = ["Non-incentive", "Incentive"]
        elif self.mode=="mc":
            labels = ["Fine","Supplies","Technical_assistance","Tax_deduction","Credit","Direct_payment"]
        else:
            labels = ["Non-incentive","Fine","Supplies","Technical_assistance","Tax_deduction","Credit","Direct_payment"]
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, display_labels=labels)
        tick_marks = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(fs, fs))
        disp.plot(ax=ax, cmap=cmap)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        plot_path = self.model_dir+f'/../CfMtx_{self.cls_mode}_{self.r}.png'
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        return plot_path
    def plot_cfmtx_om2bnmc(self):
        # get set up
        #cmap = plt.get_cmap("YlGn")
        import matplotlib.colors as colors
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap
        temp_cmap = plt.get_cmap("viridis_r")
        cmap = truncate_colormap(temp_cmap, 0.01, 0.55)
        #
        fs = 2
        labels = ["Non-incentive", "Incentive"]
        disp = ConfusionMatrixDisplay(confusion_matrix=self.om_report["bn"]["cm"], display_labels=labels)
        tick_marks = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(fs, fs))
        disp.plot(ax=ax, cmap=cmap)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        plot_path = self.model_dir+f'/../CfMtx_bn_{self.cls_mode}_{self.r}.png'
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        #
        fs = len(self.om_mc_lbs)
        #labels = ["Fine","Supplies","Technical_assistance","Tax_deduction","Credit","Direct_payment"]
        labels = [self.id2label[str(i)] for i in range(7)]#self.om_mc_lbs]
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=self.om_report["mc"]["cm"], display_labels=labels)
            tick_marks = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(fs, fs))
            disp.plot(ax=ax, cmap=cmap)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)
            plot_path = self.model_dir+f'/../CfMtx_mc_{self.cls_mode}_{self.r}.png'
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Couldn't make confusion matrix display for {self.om_report['mc']['cm']} due to {e}")
            pass
        return plot_path
    def plot_validation_loss(self):
        for el in self.metrics:
            try:
                elst = el["eval_loss"]
                self.val_loss = elst
                plt.figure()
                plt.plot(range(len(elst)), elst)
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.title(f"Validation Loss {self.r}")
                plot_path = self.model_dir+f'/../Loss_{self.r}.png'
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
            except:
                pass

class RunReporter:
    def __init__(self, run_dir, mode, ovs = False):
        self.run_dir = run_dir
        self.mode = mode
        self.run_name = run_dir.split("/")[-1]
        self.run_name = self.run_name.split("\\")[-1]
        self.run = self.run_name.split("_")[1]
        self.eval = eval
        self.models = glob.glob(self.run_dir+"/*.pt")
        self.model_reports = []
        exps = []
        for model in self.models:
            e = model.split("_")[-1][1:-3]
            exps.append(e)
        self.exps = list(set(exps))
        self.cls_mode = ""
        try:
            with open(run_dir+"/run_details.json", "r", encoding="utf-8") as f:
                self.meta_dct = json.load(f)
        except:
            with open(run_dir+"/run_details_0.json", "r", encoding="utf-8") as f:
                self.meta_dct = json.load(f)
        self.id2label = {}
        self.overall_df = None
        self.label_df_dct = None
        self.eval_batch = 32
        self.custom_acc_df = None
        self.om_df_dct = None
        self.ovs = ovs
    def load_model_reports(self, cls_mode, eval_batch=32):
        self.cls_mode = cls_mode
        self.model_reports = []
        self.eval_batch = eval_batch
        for model in tqdm(self.models):
            report = ModelReport(model, self.cls_mode, self.ovs)
            if report.mode == self.mode:
                report.load_metrics()
                report.load_config()
                randp_json = report.model_dir+f"/randp_{report.cls_mode}{'_ovs' if self.ovs else ''}.json"
                if not os.path.exists(randp_json):
                    report.calculate_randps(f"{INPUT_PTH}/ds_{report.r}_{report.mode}", self.eval_batch)
                else:
                    try:
                        with open(randp_json, "r", encoding="utf-8") as f:
                            randp = json.load(f)
                        report.load_randps(randp["real"], randp["pred"])
                    except json.JSONDecodeError as e:
                        report.calculate_randps(f"{INPUT_PTH}/ds_{report.r}_{report.mode}", self.eval_batch)
                report.calc_metrics()
                report.plot_cfmtx()
                try:
                    report.plot_validation_loss()
                except:
                    pass
                if report.mode == "om":
                    report.calc_metrics_om2bnmc()
                    report.plot_cfmtx_om2bnmc()
                self.model_reports.append(report)
                self.id2label = report.id2label
    def create_result_dfs(self):
        overall_dct = {}
        label_coll = {str(i):{exp:{} for exp in self.exps} for i in range(2 if self.mode=="bn" else 6 if self.mode=="mc" else 7)}
        cust_acc_dct = {}
        for report in self.model_reports:
            if report.mode == self.mode:
                overall_dct[report.r]=[report.cls_report["overall"][key] for key in list(report.cls_report["overall"])]
                try:
                    overall_dct[report.r].append(min(report.val_loss))
                except:
                    overall_dct[report.r].append(np.NAN)
                if self.mode in ["mc", "om"]:
                    cust_acc_dct[report.r] = [report.cls_report["custom_acc"][key] for key in list(report.cls_report["custom_acc"])]
                #print(report.cls_report["labels"])
                for label in list(report.cls_report["labels"]):
                    #print("\n",label)
                    #print(report.cls_report["labels"][label])
                    label_coll[label][report.r]=[report.cls_report["labels"][label][key] for key in list(report.cls_report["labels"][label])]
        #
        overall_df = pd.DataFrame.from_dict(overall_dct, orient='index', columns=["precision", "recall", "f1-score", "support", "accuracy", "min_vloss"])
        overall_df.loc['Average'] = overall_df.mean(skipna=True, numeric_only=True)
        overall_df.loc['SD'] = overall_df.std(skipna=True, numeric_only=True)
        self.overall_df = overall_df.drop('support', axis=1)
        #
        label_dfdct = {}
        for label in list(label_coll):
            tempdf= pd.DataFrame.from_dict(label_coll[label], orient='index', columns=["precision", "recall", "f1-score", "support"])
            tempdf = tempdf.sort_index()
            tempdf.loc["Average"] = tempdf.mean(skipna=True, numeric_only=True)
            tempdf.loc["SD"] = tempdf.std(skipna=True, numeric_only=True)
            label_dfdct[label] = tempdf.drop('support', axis=1)
        self.label_df_dct = label_dfdct
        #
        cust_acc_df = pd.DataFrame.from_dict(cust_acc_dct, orient='index', columns=["acc_wo_taxded", "taxded_acc"])
        cust_acc_df.loc['Average'] = cust_acc_df.mean(skipna=True, numeric_only=True)
        cust_acc_df.loc['SD'] = cust_acc_df.std(skipna=True, numeric_only=True)
        self.custom_acc_df = cust_acc_df
    def create_result_dfs_om2bnmc(self):
        self.om_df_dct = {"bn":{}, "mc":{}}
        omds_dct = {"bn":{}, "mc":{}}
        # bn
        # get stats
        label_coll = {str(i):{exp:{} for exp in self.exps} for i in range(2)}
        cust_acc_dct = {}
        for report in self.model_reports:
            if report.mode == self.mode:
                omds_dct["bn"][report.r]=[report.om_report["bn"]["overall"][key] for key in list(report.om_report["bn"]["overall"])]
                #print(report.om_report["bn"]["labels"])
                for label in list(report.om_report["bn"]["labels"]):
                    label_coll[str(label)][report.r]=[report.om_report["bn"]["labels"][label][key] for key in list(report.om_report["bn"]["labels"][label])]
        # overall
        overall_df = pd.DataFrame.from_dict(omds_dct["bn"], orient='index', columns=["precision", "recall", "f1-score", "support", "accuracy"])
        #print("\n"*5)
        #print(overall_df)
        overall_df.loc['Average'] = overall_df.mean(skipna=True, numeric_only=True)
        overall_df.loc['SD'] = overall_df.std(skipna=True, numeric_only=True)
        self.om_df_dct["bn"]["overall"] = overall_df.drop('support', axis=1)
        # label specific
        label_dfdct = {}
        for label in list(label_coll):
            tempdf= pd.DataFrame.from_dict(label_coll[label], orient='index', columns=["precision", "recall", "f1-score", "support"])
            tempdf = tempdf.sort_index()
            tempdf.loc["Average"] = tempdf.mean(skipna=True, numeric_only=True)
            tempdf.loc["SD"] = tempdf.std(skipna=True, numeric_only=True)
            label_dfdct[label] = tempdf.drop('support', axis=1)
        self.om_df_dct["bn"]["label_df_dct"] = label_dfdct
        #print("\n"*5)
        #for i in list(label_dfdct):
        #    print("\n",i)
        #    print(label_dfdct[i])
        ####
        #mc
        # get stats
        label_coll = {str(i):{exp:{} for exp in self.exps} for i in range(1,7)}
        cust_acc_dct = {}
        for report in self.model_reports:
            if report.mode == self.mode:
                omds_dct["mc"][report.r]=[report.om_report["mc"]["overall"][key] for key in list(report.om_report["mc"]["overall"])]
                cust_acc_dct[report.r] = [report.om_report["mc"]["custom_acc"][key] for key in list(report.om_report["mc"]["custom_acc"])]
                #print("\n",report.om_report["mc"]["labels"])
                for label in list(report.om_report["mc"]["labels"]):
                    label_coll[str(label)][report.r]=[report.om_report["mc"]["labels"][label][key] for key in list(report.om_report["mc"]["labels"][label])]
        #print("\n", omds_dct)
        # overall
        overall_df = pd.DataFrame.from_dict(omds_dct["mc"], orient='index', columns=["precision", "recall", "f1-score", "support", "accuracy"])
        overall_df.loc['Average'] = overall_df.mean(skipna=True, numeric_only=True)
        overall_df.loc['SD'] = overall_df.std(skipna=True, numeric_only=True)
        self.om_df_dct["mc"]["overall"] = overall_df.drop('support', axis=1)
        #print("\n"*5)
        #print(overall_df)
        # label specific
        label_dfdct = {}
        for label in list(label_coll):
            tempdf= pd.DataFrame.from_dict(label_coll[label], orient='index', columns=["precision", "recall", "f1-score", "support"])
            tempdf = tempdf.sort_index()
            tempdf.loc["Average"] = tempdf.mean(skipna=True, numeric_only=True)
            tempdf.loc["SD"] = tempdf.std(skipna=True, numeric_only=True)
            label_dfdct[label] = tempdf.drop('support', axis=1)
        self.om_df_dct["mc"]["label_df_dct"] = label_dfdct
        #print("\n"*5)
        #for i in list(label_dfdct):
        #    print("\n",i)
        #    print(label_dfdct[i])
        # custom acc
        cust_acc_df = pd.DataFrame.from_dict(cust_acc_dct, orient='index', columns=["acc_wo_taxded", "taxded_acc"])
        cust_acc_df.loc['Average'] = cust_acc_df.mean(skipna=True, numeric_only=True)
        cust_acc_df.loc['SD'] = cust_acc_df.std(skipna=True, numeric_only=True)
        self.om_df_dct["mc"]["custom_acc"] = cust_acc_df
    def create_label_df_html(self):
        labels_html = ""
        for label in list(self.label_df_dct):
            labels_html += f"<h4>{self.id2label[label]}</h4>"
            labels_html += self.label_df_dct[label].style.set_table_attributes('class="table"').format(precision=3).to_html()
        return labels_html
    def create_figure_display_html(self):
        figures_html = ""
        for e in sorted(self.exps):
            figures_html += f"<div class='row'><div class=col-auto>{e}</div>"
            figures_html += f"<div class=col-auto><img src='{self.run_dir}/CfMtx_{self.cls_mode}_{e}.png' alt='Confusion Matrix' width='400'></div>"
            figures_html += f"<div class=col-auto><img src='{self.run_dir}/ROC_{self.mode}_{e}.png' alt='ROC Curve' width='400'></div>"
            figures_html += f"<div class=col-auto><img src='{self.run_dir}/Loss_{e}.png' alt='Loss Curve' width='400'></div>"
            figures_html += "</div>"
        return figures_html
    def create_om2bnmc_df_html(self):
        om2bnmc_html = "<h4> Binary Aggregation </h4>"
        om2bnmc_html += self.om_df_dct["bn"]["overall"].style.set_table_attributes('class="table"').format(precision=3).to_html()
        temp = {"0":"Non-Incentive","1":"Incentive"}
        for label in list(self.om_df_dct["bn"]["label_df_dct"]):
            om2bnmc_html += f"<h5>{temp[label]}</h5>"
            om2bnmc_html += self.om_df_dct["bn"]["label_df_dct"][label].style.set_table_attributes('class="table"').format(precision=3).to_html()
        om2bnmc_html += "<h4> Muticlass Aggregation </h4>"
        om2bnmc_html += self.om_df_dct["mc"]["overall"].style.set_table_attributes('class="table"').format(precision=3).to_html()
        for label in list(self.om_df_dct["mc"]["label_df_dct"]):
            om2bnmc_html += f"<h5>{self.id2label[label]}</h5>"
            om2bnmc_html += self.om_df_dct["mc"]["label_df_dct"][label].style.set_table_attributes('class="table"').format(precision=3).to_html()
        om2bnmc_html += "<h5>Custom Accuracy</h5>"
        om2bnmc_html += self.om_df_dct["mc"]["custom_acc"].style.set_table_attributes('class="table"').format(precision=3).to_html() if self.mode in ["mc", "om"] else ""
        return om2bnmc_html
    def make_report(self, template):
        temp = Template(filename=template, strict_undefined=True)
        T = temp.render(run_name=self.run_name,
                        mode=self.mode,
                        cls_mode=self.cls_mode,
                        meta_dct =self.meta_dct,
                        overall_results = self.overall_df.style.set_table_attributes('class="table"').format(precision=3).to_html(),
                        label_results_html = self.create_label_df_html(),
                        custom_acc_res = self.custom_acc_df.style.set_table_attributes('class="table"').format(precision=3).to_html() if self.mode in ["mc", "om"] else "",
                        figures_display_html = self.create_figure_display_html(),
                        om2bnmc = self.create_om2bnmc_df_html() if self.mode == "om" else "")
        with open(os.path.join(self.run_dir, f"model_display_{self.run}_{self.mode}_{self.cls_mode}{'_ovs' if self.ovs else ''}.html"),"w") as f:
            f.write(T)
        return None

class MetaRunReporter:
    def __init__(self, meta_run_dir, mode, cls_mode, ovs = False):
        self.meta_run_dir = meta_run_dir
        self.mode = mode
        self.cls_mode = cls_mode
        self.run_collection = {}
        self.overall_df = None
        self.label_df_dct = None
        self.custom_acc_df = None
        self.id2label = {}
        self.om_df_dct = {}
        self.ovs = ovs
    def process_runs(self, report_temp = False):
        run_collection = {}
        dir_names = glob.glob(self.meta_run_dir+f"/*{self.mode}")
        for dn in tqdm(dir_names):
            if os.path.isdir(dn):
                print("\n",dn.split("/")[-1])
                rr = RunReporter(dn, self.mode)
                try:
                    rr.load_model_reports(self.cls_mode)# or "svm"
                    rr.create_result_dfs()
                    if rr.mode == "om":
                        rr.create_result_dfs_om2bnmc()
                    run_collection[rr.run] = rr
                    if rr.id2label:
                        self.id2label = rr.id2label
                    if report_temp:
                        rr.make_report(report_temp)
                except Exception as e:
                    print(f"Could not load {rr.run} {self.mode}", dn.split("/")[-1], "due to", e)
                    pass
        self.run_collection = run_collection
    def generate_overall_df(self):
        overall_df = pd.DataFrame(index = list(self.run_collection), columns=["precision", "recall", "f1-score", "accuracy"])
        for rn in list(self.run_collection):
            rreport = self.run_collection[rn]
            report_df = rreport.overall_df
            for metric in report_df.columns:
                overall_df.loc[rreport.run, metric]=report_df[metric]["Average"]
        self.overall_df = overall_df
    def generate_label_df(self):
        label_dfdct = {str(i):pd.DataFrame(index = list(self.run_collection), columns=["precision", "recall", "f1-score"]) for i in range(2 if self.mode=="bn" else 6 if self.mode=="mc" else 7)}
        for rn in list(self.run_collection):
            rreport = self.run_collection[rn]
            report_df_dct = rreport.label_df_dct
            #print("\n",rn,"\n")
            for label in list(report_df_dct):
                for metric in report_df_dct[label].columns:
                    #print(label, "\n", list(label_dfdct),"\n", list(report_df_dct))
                    label_dfdct[label].loc[rreport.run, metric] = report_df_dct[label][metric]["Average"]
        self.label_df_dct = label_dfdct
    def generate_custom_acc_df(self):
        custom_acc_df = pd.DataFrame(index = list(self.run_collection), columns=["acc_wo_taxded", "taxded_acc"])
        for rn in list(self.run_collection):
            rreport = self.run_collection[rn]
            report_df = rreport.custom_acc_df
            for metric in report_df.columns:
                custom_acc_df.loc[rreport.run, metric]=report_df[metric]["Average"]
        self.custom_acc_df = custom_acc_df
    def generate_om2bnmc_dfs(self):
        self.om_df_dct = {"bn":{}, "mc":{}}
        # bn
        # overall
        overall_df = pd.DataFrame(index = list(self.run_collection), columns=["precision", "recall", "f1-score", "accuracy"])
        for rn in list(self.run_collection):
            rreport = self.run_collection[rn]
            report_df = rreport.om_df_dct["bn"]["overall"]
            for metric in report_df.columns:
                overall_df.loc[rreport.run, metric]=report_df[metric]["Average"]
        self.om_df_dct["bn"]["overall"] = overall_df
        # label
        label_dfdct = {str(i):pd.DataFrame(index = list(self.run_collection), columns=["precision", "recall", "f1-score"]) for i in range(2 if self.mode=="bn" else 6 if self.mode=="mc" else 7)}
        for rn in list(self.run_collection):
            rreport = self.run_collection[rn]
            report_df_dct = rreport.om_df_dct["bn"]["label_df_dct"]
            for label in list(report_df_dct):
                for metric in report_df_dct[label].columns:
                    label_dfdct[label].loc[rreport.run, metric] = report_df_dct[label][metric]["Average"]
        self.om_df_dct["bn"]["label_df_dct"] = label_dfdct
        # mc
        # overall
        overall_df = pd.DataFrame(index = list(self.run_collection), columns=["precision", "recall", "f1-score", "accuracy"])
        for rn in list(self.run_collection):
            rreport = self.run_collection[rn]
            report_df = rreport.om_df_dct["mc"]["overall"]
            for metric in report_df.columns:
                overall_df.loc[rreport.run, metric]=report_df[metric]["Average"]
        self.om_df_dct["mc"]["overall"] = overall_df
        # label
        label_dfdct = {str(i):pd.DataFrame(index = list(self.run_collection), columns=["precision", "recall", "f1-score"]) for i in range(2 if self.mode=="bn" else 6 if self.mode=="mc" else 7)}
        for rn in list(self.run_collection):
            rreport = self.run_collection[rn]
            report_df_dct = rreport.om_df_dct["mc"]["label_df_dct"]
            for label in list(report_df_dct):
                for metric in report_df_dct[label].columns:
                    label_dfdct[label].loc[rreport.run, metric] = report_df_dct[label][metric]["Average"]
        self.om_df_dct["mc"]["label_df_dct"] = label_dfdct
        # custom acc
        custom_acc_df = pd.DataFrame(index = list(self.run_collection), columns=["acc_wo_taxded", "taxded_acc"])
        for rn in list(self.run_collection):
            rreport = self.run_collection[rn]
            report_df = rreport.om_df_dct["mc"]["custom_acc"]
            for metric in report_df.columns:
                custom_acc_df.loc[rreport.run, metric]=report_df[metric]["Average"]
        self.om_df_dct["mc"]["custom_acc"] = custom_acc_df
    def create_om2bnmc_df_html(self):
        om2bnmc_html = "<h4> Binary Aggregation </h4>"
        om2bnmc_html += self.om_df_dct["bn"]["overall"].style.set_table_attributes('class="table"').format(precision=3).to_html()
        for label in list(self.om_df_dct["bn"]["label_df_dct"]):
            om2bnmc_html += f"<h5>{self.id2label[label]}</h5>"
            om2bnmc_html += self.om_df_dct["bn"]["label_df_dct"][label].style.set_table_attributes('class="table"').format(precision=3).to_html()
        om2bnmc_html += "<h4> Muticlass Aggregation </h4>"
        om2bnmc_html += self.om_df_dct["mc"]["overall"].style.set_table_attributes('class="table"').format(precision=3).to_html()
        for label in list(self.om_df_dct["mc"]["label_df_dct"]):
            om2bnmc_html += f"<h5>{self.id2label[label]}</h5>"
            om2bnmc_html += self.om_df_dct["mc"]["label_df_dct"][label].style.set_table_attributes('class="table"').format(precision=3).to_html()
        om2bnmc_html += "<h5>Custom Accuracy</h5>"
        om2bnmc_html += self.om_df_dct["mc"]["custom_acc"].style.set_table_attributes('class="table"').format(precision=3).to_html() if self.mode in ["mc", "om"] else ""
        return om2bnmc_html
    def create_label_df_html(self):
        labels_html = ""
        for label in list(self.label_df_dct):
            labels_html += f"<h4>{self.id2label[label]}</h4>"
            labels_html += self.label_df_dct[label].style.set_table_attributes('class="table"').format(precision=3).to_html()
        return labels_html
    def make_report(self, template):
        temp = Template(filename=template, strict_undefined=True)
        T = temp.render(mode=self.mode,
                        cls_mode=self.cls_mode,
                        overall_results = self.overall_df.style.set_table_attributes('class="table"').format(precision=3).to_html(),
                        label_results_html =self.create_label_df_html(),
                        custom_acc_res = self.custom_acc_df.style.set_table_attributes('class="table"').format(precision=3).to_html() if self.mode =="mc" else "",
                        figures_display_html="None",
                        om2bnmc = self.create_om2bnmc_df_html() if self.mode == "om" else "")
        with open(os.path.join(self.meta_run_dir, f"MetaDisplay_{self.mode}_{self.cls_mode}{'_ovs' if self.ovs else ''}.html"),"w") as f:
            f.write(T)
        return None
    
def create_run_report(run_dir, mode, cls_mode, ovs=False):
    rr = RunReporter(run_dir, mode, ovs)
    rr.load_model_reports(cls_mode)# or "svm"
    rr.create_result_dfs()
    if mode=="om":
        rr.create_result_dfs_om2bnmc()
    rr.make_report(f"{CWD}/classifier/runrpt_template.html")

def create_meta_report(meta_dir, mode, cls_mode, ovs=False):
    mrr = MetaRunReporter(meta_dir, mode, cls_mode, ovs)
    mrr.process_runs(f"{CWD}/classifier/runrpt_template.html")
    mrr.generate_overall_df()
    mrr.generate_label_df()
    if mode in ["mc","om"]:
        mrr.generate_custom_acc_df()
        if mode=="om":
            mrr.generate_om2bnmc_dfs()
    mrr.make_report(f"{CWD}/classifier/meta_runrpt_template.html")

if __name__ == "__main__":
    cwd = os.getcwd()
    odir = cwd+"/../outputs"
    idir = cwd+"/../inputs"
    '''
    for mode in ["bn", "mc"]:#
        for cls_mode in ["rf"]:#"model", "svm"
            create_meta_report(odir, mode, cls_mode)
    '''
    #create_run_report(odir+"/fting_L_bn", "bn", "svm")
    #create_run_report(odir+"/fting_M_mc", "mc", "svm")
    odir="E:/PhD/2June2025"
    create_meta_report(odir, "bn", "svm")
    create_meta_report(odir, "mc", "svm")
    create_meta_report(odir, "bn", "model")
    create_meta_report(odir, "mc", "model")