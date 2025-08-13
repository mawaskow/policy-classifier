This repository presents the workflow for the Irish implementation of [this ML pipeline for policy analysis](https://www.sciencedirect.com/science/article/pii/S1389934121002306), adapted and updated from [this repository](https://github.com/wri-dssg-omdena/policy-data-analyzer).

The final dataset produced through this pipeline can be found [here](https://huggingface.co/datasets/mawaskow/irish_forestry_incentives), complete with a dataset card for transparency and reproducibility.

#### About

This code demonstrates the full pipeline of our workshop paper (submitted, under review) aimed at demonstrating the applicability of the original pipeline, used on Spanish-language forestry policies in 5 Latin American countries, to English-language forestry policies in Ireland. We gather policies with a scraper, extract cleaned sentences and label them for (binary) containing a policy incentive and (multiclass) the kind of policy incentive they contain, then train both a binary and a multiclass sentence classifier on the dataset.

#### Repo Structure
- inputs : contains the json datasets of the hand-labeled, HITL-labeled, and validation datasets
- policy_scraping : contains the scrapy spider used to gather the policies from the website of the Irish government, including the keywords used to search for policies and a zip folder of the original policies found (which will need to be unzipped if you want to follow the pipeline from that point)
- populate_corpora : contains the code to preprocess the PDFs and prepare the corpora (including the data augmentation and exernal annotator/validator section)
- classifier : contains the code to fine-tune binary and multiclass classification models on the dataset, as well as the templates to create the comparative reports of model performance 

#### Use
- *workflow.ipynb* : demonstrates the pipeline across its many functions-- this is a good starting point to understand the repository.