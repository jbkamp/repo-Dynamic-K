### Welcome
This is the repository of the short paper [Dynamic Top-K Estimation Consolidates Disagreement between Feature 
Attribution Methods](https://arxiv.org/abs/2310.05619) by Jonathan Kamp, Lisa Beinborn and Antske Fokkens. EMNLP 2023, 
Singapore.

### Description of the code files in `esnli/` directory
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
Please cite our EMNLP version:
`
@inproceedings{kamp-etal-2023-dynamic,
    title = "Dynamic Top-k Estimation Consolidates Disagreement between Feature Attribution Methods",
    author = "Kamp, Jonathan  and
      Beinborn, Lisa  and
      Fokkens, Antske",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.379",
    doi = "10.18653/v1/2023.emnlp-main.379",
    pages = "6190--6197",
    abstract = "Feature attribution scores are used for explaining the prediction of a text classifier to users by highlighting a k number of tokens. In this work, we propose a way to determine the number of optimal k tokens that should be displayed from sequential properties of the attribution scores. Our approach is dynamic across sentences, method-agnostic, and deals with sentence length bias. We compare agreement between multiple methods and humans on an NLI task, using fixed k and dynamic k. We find that perturbation-based methods and Vanilla Gradient exhibit highest agreement on most method{--}method and method{--}human agreement metrics with a static k. Their advantage over other methods disappears with dynamic ks which mainly improve Integrated Gradient and GradientXInput. To our knowledge, this is the first evidence that sequential properties of attribution scores are informative for consolidating attribution signals for human interpretation.",
}
`
