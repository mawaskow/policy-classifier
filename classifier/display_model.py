from mako.template import Template
import json
import os
import glob, regex
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from tqdm import tqdm

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
        self.cm = confusion_matrix(self.real, self.pred, normalize="true")
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
    def load_model_reports(self, cls_mode):
        self.cls_mode = cls_mode
        self.model_reports = []
        for model in tqdm(self.models):
            report = ModelReport(model, self.cls_mode)
            if report.mode == self.mode:
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
                self.id2label = report.id2label
    def create_result_dfs(self):
        overall_dct = {}
        label_coll = {str(i):{exp:{} for exp in self.exps} for i in range(2 if self.mode=="bn" else 6)}
        for report in self.model_reports:
            if report.mode == self.mode:
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
        self.id2label = {}
    def process_runs(self, report_temp = False):
        run_collection = {}
        #dir_names = glob.glob(self.meta_run_dir+"/*")
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
                    print(f"Could not do {rr.run} {self.mode}", dn.split("/")[-1], "due to", e)
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
                        figures_display_html="None")
        with open(os.path.join(self.meta_run_dir, f"MetaDisplay_{self.mode}_{self.cls_mode}.html"),"w") as f:
            f.write(T)
        return None

if __name__ == "__main__":
    cwd = os.getcwd()
    odir = cwd+"/../outputs"
    '''
    dir_names = glob.glob(odir+"/*")
    #dir_names = ['.\\fting_G_A_again', '.\\fting_H_0CELwght', '.\\fting_I_D_again', '.\\fting_J_wghtCEL3', '.\\fting_K_lr1e5_wght', '.\\fting_L_oversamplingauto', '.\\fting_M_os_lr2e6', '.\\fting_N_os_bnlr2e6', '.\\fting_O_bn_5e6']
    #dir_names = ['.\\fting_M_os_lr2e6']
    for dn in dir_names:
        print("\n",dn.split("/")[-1])
        #dn = odir+"/"+dn
        rr = RunReporter(dn)
        #
        try:
            rr.load_model_reports("model")# or "svm"
            rr.create_result_dfs("bn")
            rr.make_report(cwd+"/runrpt_template.html", "bn")
        except Exception as e:
            print("Could not do BN", dn.split("/")[-1], "due to", e)
            pass
        #
        try:
            rr.load_model_reports("model")# or "svm"
            rr.create_result_dfs("mc")
            rr.make_report(cwd+"/runrpt_template.html", "mc")
        except Exception as e:
            print("Could not do MC", dn.split("/")[-1], "due to", e)
            pass
    '''
    for i in ["bn", "mc"]:
        for j in ["model", "svm"]:
            mrr = MetaRunReporter(odir, i, j)
            mrr.process_runs(cwd+"/runrpt_template.html")
            mrr.generate_overall_df()
            #print(mrr.overall_df)
            mrr.generate_label_df()
            #for label in list(mrr.label_df_dct):
            #    print("\n", label)
            #    print(mrr.label_df_dct[label])
            mrr.make_report(cwd+"/meta_runrpt_template.html")
    print("Done.")