This repository presents the workflow for the Irish implementation of [this ML pipeline for policy analysis](https://www.sciencedirect.com/science/article/pii/S1389934121002306), adapted and updated from [this repository](https://github.com/wri-dssg-omdena/policy-data-analyzer).

The final dataset produced through this pipeline can be found [here](https://huggingface.co/datasets/mawaskow/irish_forestry_incentives), as well as the [binary](https://huggingface.co/mawaskow/inc_sent_cls_bn) and [mutliclass](https://huggingface.co/mawaskow/inc_sent_cls_mc) classification models.

#### About

This code demonstrates the full pipeline of our [workshop paper](https://aclanthology.org/2025.konvens-2.6) aimed at demonstrating the applicability of the original pipeline, used on Spanish-language forestry policies in 5 Latin American countries, to English-language forestry policies in Ireland. We gather policies with a scraper, extract cleaned sentences and label them for (binary) containing a policy incentive and (multiclass) the kind of policy incentive they contain, then train both a binary and a multiclass sentence classifier on the dataset.

#### Repo Structure
- inputs : contains the json datasets of the hand-labeled, HITL-labeled, and validation datasets
- policy_scraping : contains the scrapy spider used to gather the policies from the website of the Irish government, including the keywords used to search for policies and a [zip folder](https://github.com/mawaskow/policy-classifier/blob/main/policy_scraping/policy_scraping/outputs/forestry/full.zip) of the original policies found (which will need to be unzipped for use in the pipeline)
- populate_corpora : contains the code to preprocess the PDFs and prepare the corpora (including the data augmentation and exernal annotator/validator section)
- classifier : contains the code to fine-tune binary and multiclass classification models on the dataset, as well as the templates to create the comparative reports of model performance

#### Use
- *workflow.ipynb* : demonstrates the pipeline across its many functions-- this is a good starting point to understand the repository

#### How to Cite
@inproceedings{waskow2025enhancing,
  title={Enhancing Policy Analysis with NLP: A Reproducible Approach to Incentive Classification},
  author={Waskow, MA and McCrae, John Philip},
  booktitle={Proceedings of the 21st Conference on Natural Language Processing (KONVENS 2025): Workshops},
  pages={74--85},
  year={2025}
}
