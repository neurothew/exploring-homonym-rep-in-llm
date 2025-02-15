# exploring-homonym-representations-in-llm
This repository is the official implementation of the following paper:

Anonymous (2025). Exploring Layer-wise Representations of English and Chinese Homonymy in Pre-trained Language Models.

## Folder structure

```
code/
├── data/                        # data for computation analyses
│   └── df_homonymy_v20251117.xlsx                 
├── scripts/                     # scripts for computing homonym representations                  
│   ├── run_compute_lexical_ambiguity_v20251019.sh
│   ├── run_compute_homonym_baseline_v20251018.sh
statistics/                    
├── data/                        # data for statistical analyses                  
│   └── df_sim_byword_preprocessed.feather                    
├── figures/                     # storing the figures on manuscript
├── results/                  
│   ├── df_en_model_diff_on_best_dscores.csv
│   ├── df_zh_model_diff_on_best_dscores.csv
│   └── df_pairs_pos_all.csv
└── scripts/                      # scripts for statistical analyses
    └── analysis_homonymy.Rmd
```

`df_homonymy_v20251117.xlsx`
- The dataframe storing all the homonyms and sentences.

`run_compute_homonym_baseline_v20251018.sh`
- The shell script to compute angular similarity baseline

`run_compute_lexical_ambiguity_v20251019.sh`
- The shell script to compute angular similarity between homonyms.
=
`analysis_homonymy.Rmd`
- script to generate all the statistical analyses, tables and figures.

`df_pairs_pos_all.csv`
- The dataframe storing the pairwise comparisons between same-POS and diff-POS D-score.

`df_en/zh_model_diff_on_best_dscores.csv`
- The dataframe storing the pairwise comparisons of the best D-scores computed from English and Chinese homonym representations from each model.





