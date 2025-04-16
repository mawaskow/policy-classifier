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

def encode_all_sents(all_sents, sbert_model):
    '''
    modified from previous repository's latent_embeddings_classifier.py
    '''
    stacked = np.vstack([sbert_model.encode(sent) for sent in tqdm(all_sents)])
    return [torch.from_numpy(element).reshape((1, element.shape[0])) for element in stacked]

class ModelReport:
    def __init__(self, model_dir, cls_mode="model"):
        self.model_dir = model_dir
        self.cls_mode=cls_mode
        self.model_name = model_dir.split("/")[-1][:-3]
        self.model_name = self.model_name.split("\\")[-1]
        self.metrics = self.load_metrics()
        self.config = self.load_config()
        self.id2label = self.config["id2label"]
        self.label2id = self.config["label2id"]
        self.real = None
        self.predicted = None
        self.name_dct = {
            "paraphrase-xlm-r-multilingual-v1":"bert"
        }
        model_type = self.model_name.split("_")[0]
        self.callname = self.model_name.replace(model_type, self.name_dct[model_type])
        self.mode = self.callname.split("_")[1]
        self.r = self.callname.split("_")[-1][1:]
        self.cls_report = {}
        self.cm = []
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
            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, num_labels=num_lbs,id2label=self.id2label, label2id=self.label2id).to(dev)
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
            model = SentenceTransformer(self.model_dir, device=dev)
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
        with open(self.model_dir+f"/randp_{self.cls_mode}.json", "w", encoding="utf-8") as f:
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
        if self.mode == "mc":
            self.cls_report["custom_acc"] = {}
            real_ar = np.array(self.real)
            pred_ar = np.array(self.pred)
            mask = real_ar != 3
            notd_real = real_ar[mask]
            notd_pred = pred_ar[mask]
            acc_no_td = accuracy_score(notd_real, notd_pred)
            self.cls_report["custom_acc"]["without_td"] = acc_no_td
            #
            td_mask = real_ar == 3
            td_real = real_ar[td_mask]
            td_pred = pred_ar[td_mask]
            acc_td = accuracy_score(td_real, td_pred)
            self.cls_report["custom_acc"]["only_td"] = acc_td
            #print(self.cls_report["custom_acc"])
    def plot_cfmtx(self):
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm)
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax)
        plot_path = self.model_dir+f'/../CfMtx_{self.mode}_{self.r}.png'
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        return plot_path

class RunReporter:
    def __init__(self, run_dir, mode):
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
        with open(run_dir+"/run_details.json", "r", encoding="utf-8") as f:
            self.meta_dct = json.load(f)
        self.id2label = {}
        self.overall_df = None
        self.label_df_dct = None
        self.eval_batch = 32
        self.custom_acc_df = None
    def load_model_reports(self, cls_mode, eval_batch=32):
        self.cls_mode = cls_mode
        self.model_reports = []
        self.eval_batch = eval_batch
        for model in tqdm(self.models):
            report = ModelReport(model, self.cls_mode)
            if report.mode == self.mode:
                report.load_metrics()
                report.load_config()
                randp_json = report.model_dir+f"/randp_{report.cls_mode}.json"
                if not os.path.exists(randp_json):
                    report.calculate_randps(self.run_dir+f"/../../inputs/ds_{report.r}_{report.mode}", self.eval_batch)
                else:
                    try:
                        with open(randp_json, "r", encoding="utf-8") as f:
                            randp = json.load(f)
                        report.load_randps(randp["real"], randp["pred"])
                    except json.JSONDecodeError as e:
                        report.calculate_randps(self.run_dir+f"/../../inputs/ds_{report.r}_{report.mode}", self.eval_batch)
                report.calc_metrics()
                report.plot_cfmtx()
                self.model_reports.append(report)
                self.id2label = report.id2label
    def create_result_dfs(self):
        overall_dct = {}
        label_coll = {str(i):{exp:{} for exp in self.exps} for i in range(2 if self.mode=="bn" else 6)}
        cust_acc_dct = {}
        for report in self.model_reports:
            if report.mode == self.mode:
                overall_dct[report.r]=[report.cls_report["overall"][key] for key in list(report.cls_report["overall"])]
                if self.mode =="mc":
                    cust_acc_dct[report.r] = [report.cls_report["custom_acc"][key] for key in list(report.cls_report["custom_acc"])]
                for label in list(report.cls_report["labels"]):
                    label_coll[label][report.r]=[report.cls_report["labels"][label][key] for key in list(report.cls_report["labels"][label])]
        #
        overall_df = pd.DataFrame.from_dict(overall_dct, orient='index', columns=["precision", "recall", "f1-score", "support", "accuracy"])
        overall_df.loc['Average'] = overall_df.mean(numeric_only=True)
        self.overall_df = overall_df.drop('support', axis=1)
        #
        label_dfdct = {}
        for label in list(label_coll):
            tempdf= pd.DataFrame.from_dict(label_coll[label], orient='index', columns=["precision", "recall", "f1-score", "support"])
            tempdf = tempdf.sort_index()
            tempdf.loc["Average"] = tempdf.mean(numeric_only=True)
            label_dfdct[label] = tempdf.drop('support', axis=1)
        self.label_df_dct = label_dfdct
        #
        cust_acc_df = pd.DataFrame.from_dict(cust_acc_dct, orient='index', columns=["acc_wo_taxded", "taxded_acc"])
        cust_acc_df.loc['Average'] = cust_acc_df.mean(numeric_only=True)
        self.custom_acc_df = cust_acc_df
    def create_label_df_html(self):
        labels_html = ""
        for label in list(self.label_df_dct):
            labels_html += f"<h4>{self.id2label[label] if self.mode=='bn' else self.id2label[label]}</h4>"
            labels_html += self.label_df_dct[label].style.set_table_attributes('class="table"').format(precision=3).to_html()
        return labels_html
    def create_figure_display_html(self):
        figures_html = ""
        for e in self.exps:
            figures_html += f"<div class='row'><div class=col-auto>{e}</div>"
            figures_html += f"<div class=col-auto><img src='{self.run_dir}/CfMtx_{self.mode}_{e}.png' alt='Confusion Matrix' width='400'></div>"
            figures_html += f"<div class=col-auto><img src='{self.run_dir}/ROC_{self.mode}_{e}.png' alt='ROC Curve' width='400'></div>"
            figures_html += "</div>"
        return figures_html
    def make_report(self, template):
        temp = Template(filename=template, strict_undefined=True)
        T = temp.render(run_name=self.run_name,
                        mode=self.mode,
                        cls_mode=self.cls_mode,
                        meta_dct =self.meta_dct,
                        overall_results = self.overall_df.style.set_table_attributes('class="table"').format(precision=3).to_html(),
                        label_results_html = self.create_label_df_html(),
                        custom_acc_res = self.custom_acc_df.style.set_table_attributes('class="table"').format(precision=3).to_html() if self.mode =="mc" else "",
                        figures_display_html = self.create_figure_display_html())
        with open(os.path.join(self.run_dir, f"model_display_{self.run}_{self.mode}_{self.cls_mode}.html"),"w") as f:
            f.write(T)
        return None

class MetaRunReporter:
    def __init__(self, meta_run_dir, mode, cls_mode):
        self.meta_run_dir = meta_run_dir
        self.mode = mode
        self.cls_mode = cls_mode
        self.run_collection = {}
        self.overall_df = None
        self.label_df_dct = None
        self.custom_acc_df = None
        self.id2label = {}
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
        label_dfdct = {str(i):pd.DataFrame(index = list(self.run_collection), columns=["precision", "recall", "f1-score"]) for i in range(2 if self.mode=="bn" else 6)}
        for rn in list(self.run_collection):
            rreport = self.run_collection[rn]
            report_df_dct = rreport.label_df_dct
            for label in list(report_df_dct):
                for metric in report_df_dct[label].columns:
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
                        figures_display_html="None")
        with open(os.path.join(self.meta_run_dir, f"MetaDisplay_{self.mode}_{self.cls_mode}.html"),"w") as f:
            f.write(T)
        return None
    
def create_run_report(run_dir, mode, cls_mode):
    rr = RunReporter(run_dir, mode)
    rr.load_model_reports(cls_mode)# or "svm"
    rr.create_result_dfs()
    rr.make_report("./runrpt_template.html")

def create_meta_report(meta_dir, mode, cls_mode):
    mrr = MetaRunReporter(meta_dir, mode, cls_mode)
    mrr.process_runs("./runrpt_template.html")
    mrr.generate_overall_df()
    mrr.generate_label_df()
    if mode =="mc":
        mrr.generate_custom_acc_df()
    mrr.make_report("./meta_runrpt_template.html")

if __name__ == "__main__":
    cwd = os.getcwd()
    odir = cwd+"/../outputs"
    idir = cwd+"/../inputs"
    '''
    create_run_report(odir+"/fting_A_bn", "bn", "rf")
    create_run_report(odir+"/fting_B_mc", "mc", "rf")

    '''
    for mode in ["bn", "mc"]:#
        for cls_mode in ["rf"]:#"model", "svm"
            create_meta_report(odir, mode, cls_mode)
    
