from mako.template import Template
import json
import os
import glob, regex
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

class ModelReport:
    def __init__(self, model_dir, cls_mode="model"):
        self.model_dir = model_dir
        self.cls_mode=cls_mode
        self.model_name = model_dir.split("/")[-1][:-3]
        self.model_name = self.model_name.split("\\")[-1]
        self.metrics = self.load_metrics()
        self.config = self.load_config()
        self.id2label = self.config["id2label"]
        self.real = None
        self.predicted = None
        name_dct = {
            "paraphrase-xlm-r-multilingual-v1":"bert"
        }
        model_type = self.model_name.split("_")[0]
        self.callname = self.model_name.replace(model_type, name_dct[model_type])
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
        self.cm = confusion_matrix(self.real, self.pred)
    def plot_cfmtx(self):
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm)
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax)
        plot_path = self.model_dir+f'/../CfMtx_{self.mode}_{self.r}.png'
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        return plot_path

class RunReporter:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.run_name = run_dir.split("/")[-1]
        self.run_name = self.run_name.split("\\")[-1]
        self.eval = eval
        self.models = glob.glob(self.run_dir+"/*.pt")
        self.model_reports = []
        exps = []
        bn_mdls = []
        mc_mdls = []
        for model in self.models:
            if model.split("_")[1]== "bn":
                bn_mdls.append(model)
            else:
                mc_mdls.append(model)
            e = model.split("_")[-1][1:-3]
            exps.append(e)
        self.bn_models = bn_mdls
        self.mc_models = mc_mdls
        self.exps = list(set(exps))
        self.cls_mode = ""
        with open(run_dir+"/run_details.json", "r", encoding="utf-8") as f:
            self.meta_dct = json.load(f)
        self.id2label_bn = {}
        self.id2label_mc = {}
        self.overall_df = None
        self.label_df_dct = None
    def load_model_reports(self, cls_mode):
        self.cls_mode = cls_mode
        self.model_reports = []
        for model in self.models:
            report = ModelReport(model, self.cls_mode)
            report.load_metrics()
            report.load_config()
            rnp_json = glob.glob(self.run_dir+f"/randp_*{report.cls_mode}.json")[0]
            with open(rnp_json) as f:
                self.rnp = json.load(f)
            real = self.rnp[report.mode][report.callname]["real"]
            pred = self.rnp[report.mode][report.callname]["pred"]
            report.load_randps(real, pred)
            report.calc_metrics()
            report.plot_cfmtx()
            self.model_reports.append(report)
            if report.mode == "bn":
                self.id2label_bn = report.id2label
            else:
                self.id2label_mc = report.id2label
    def create_result_dfs(self, mode):
        overall_dct = {}
        label_coll = {str(i):{exp:{} for exp in self.exps} for i in range(2 if mode=="bn" else 6)}
        for report in self.model_reports:
            if report.mode == mode:
                overall_dct[report.r]=[report.cls_report["overall"][key] for key in list(report.cls_report["overall"])]
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
    def create_label_df_html(self, mode):
        labels_html = ""
        for label in list(self.label_df_dct):
            labels_html += f"<h4>{self.id2label_bn[label] if mode=='bn' else self.id2label_mc[label]}</h4>"
            labels_html += self.label_df_dct[label].style.set_table_attributes('class="styled-table"').format(precision=3).to_html()
        return labels_html
    def create_figure_display_html(self, mode):
        figures_html = ""
        for e in self.exps:
            figures_html += f"<div class='row'><div class=col-auto>{e}</div>"
            figures_html += f"<div class=col-auto><img src='{self.run_dir}/CfMtx_{mode}_{e}.png' alt='Confusion Matrix' width='400'></div>"
            figures_html += f"<div class=col-auto><img src='{self.run_dir}/ROC_{mode}_{e}.png' alt='ROC Curve' width='400'></div>"
            figures_html += "</div>"
        return figures_html
    def make_report(self, template, mode):
        temp = Template(filename=template, strict_undefined=True)
        T = temp.render(run_name=self.run_name,
                        mode=mode,
                        cls_mode=self.cls_mode,
                        meta_dct =self.meta_dct,
                        overall_results = self.overall_df.style.set_table_attributes('class="styled-table"').format(precision=3).to_html(),
                        label_results_html = self.create_label_df_html(mode),
                        figures_display_html = self.create_figure_display_html(mode))
        with open(os.path.join(self.run_dir, f"model_display_{mode}_{self.cls_mode}.html"),"w") as f:
            f.write(T)
        return None

if __name__ == "__main__":
    cwd = os.getcwd()
    odir = cwd+"/../outputs"
    dir_names = glob.glob(odir+"/*")
    for dn in dir_names:
        print("\n",dn.split("/")[-1])
        rr = RunReporter(dn)
        #
        try:
            rr.load_model_reports("model")# or "svm"
            rr.create_result_dfs("bn")
            rr.make_report(cwd+"/model_template.html", "bn")
        except Exception as e:
            print("Could not do", dn.split("/")[-1], "due to", e)
            pass
        #
        try:
            rr.load_model_reports("model")# or "svm"
            rr.create_result_dfs("mc")
            rr.make_report(cwd+"/model_template.html", "mc")
        except Exception as e:
            print("Could not do", dn.split("/")[-1], "due to", e)
            pass
    #rr.make_mc_report(cwd+"/model_template.html")
    print("Done.")