### Welcome
This is the repository of the short paper [Dynamic Top-K Estimation Consolidates Disagreement between Feature 
Attribution Methods](https://arxiv.org/abs/2310.05619) by Jonathan Kamp, Lisa Beinborn and Antske Fokkens. EMNLP 2023, 
Singapore.

### Description of the code files in `esnli/`
Fine-tuning the models:
* `esnli_distilbert.py`

Results inspection and stats:
* `esnli_distilbert_evalonly.py`
* `esnli_dataset_and_annotation_stats.py`

Model selection:
* `create_explanation_pickles_distilbert.py` (stored in `explanations/`)
* `classifier_model_selection.py`

Analysis script:
* `esnli_analysis_distilbert.py`

### Dependencies
* `Python 3.9.13`
* `torch 1.14.0.dev20221104`
* `transformers 4.24.0`
* `ferret 0.4.1`


### Cite us :)
Please cite our EMNLP reference when available.



