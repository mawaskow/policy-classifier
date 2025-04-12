from mako.template import Template
import json
import os
import glob, regex
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ModelReport:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model_name = model_dir.split("/")[-1][:-3]
        self.model_name = self.model_name.split("\\")[-1]
        self.metrics = self.load_metrics()
        self.config = self.load_config()
        self.real = None
        self.predicted = None
        name_dct = {
            "paraphrase-xlm-r-multilingual-v1":"bert"
        }
        model_type = self.model_name.split("_")[0]
        self.callname = self.model_name.replace(model_type, name_dct[model_type])
        self.mode = self.callname.split("_")[1]
        self.r = self.callname.split("_")[-1][1:]
        self.overall_model_metrics = {}
        self.label_model_metrics = {}
        self.overall_svm_metrics = {}
        self.label_svm_metrics = {}
    def load_metrics(self):
        metrics_path = self.model_dir+'/metrics.json'
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except:
            return {}
    def load_config(self):
        config_path = self.model_dir+'/config.json'
        try:
            with open(config_path) as f:
                return json.load(f)
        except:
            return {}
    def load_randps(self, real, pred):
        self.real = real
        self.pred = pred
    def calculate_metrics(self):
        return None
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.real, self.predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax)
        plot_path = self.model_dir+f'/../CfMtx_{self.mode}_{self.r}.png'
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        return plot_path

class RunReporter():
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.models = glob.glob(self.run_dir+"/*.pt")
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
        rnp_mdl_json = glob.glob(self.run_dir+"/randp_*model.json")[0]
        rnp_svm_json = glob.glob(self.run_dir+"/randp_*svm.json")[0]
        with open(rnp_mdl_json) as f:
            self.rnp_mdls = json.load(f)
        with open(rnp_svm_json) as f:
            self.rnp_svms = json.load(f)
        with open(run_dir+"/run_details.json", "r", encoding="utf-8") as f:
            self.meta_dct = json.load(f)
        self.model_reports = []
        for model in self.models:
            report = ModelReport(model)
            report.load_metrics()
            report.load_config()
            real = self.rnp_mdls[report.mode][report.callname]["real"]
            pred = self.rnp_mdls[report.mode][report.callname]["pred"]
            report.load_randps(real, pred)
            self.model_reports.append(report)
    def make_bn_report(self, template):
        mode = "bn"
        temp = Template(filename=template, strict_undefined=True)
        T = temp.render(run_name=self.run_dir,
                        mode="BN",
                        batch_size=self.meta_dct[mode]["batch_size"])
        with open(os.path.join(self.run_dir, "model_display_BN.html"),"w") as f:
            f.write(T)
        return None
    def make_mc_report(self, template):
        mode = "mc"
        temp = Template(filename=template, strict_undefined=True)
        T = temp.render(run_name=self.run_dir,
                        mode="MC",
                        batch_size=self.meta_dct[mode]["batch_size"])
        with open(os.path.join(self.run_dir, "model_display_MC.html"),"w") as f:
            f.write(T)
        return None

if __name__ == "__main__":
    cwd = os.getcwd()
    odir = cwd+"/../outputs"
    rr = RunReporter(odir+"/fting")
    print(rr.model_reports)
    rr.make_bn_report(cwd+"/model_template.html")
    #rr.make_mc_report(cwd+"/model_template.html")
    print("Done.")