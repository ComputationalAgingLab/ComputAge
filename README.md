![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
![made with love](https://img.shields.io/badge/made%20with%20%E2%9D%A4%EF%B8%8F-8A2BE2)

# ComputAge
A library for full-stack aging clocks design and benchmarking.

*The full release version of this package is currently under development. Only the bechmarking module is released and ready for use. Please see below.*

## Installation

You can install the whole library with `pip`:

`pip install computage`

This provides all instruments necessary for aging clocks benchmarking.

# ComputAgeBench

A module in the `computage` library for epigenetic aging clocks benchmarking. This library is tightly bound with the `computage_bench` Hugging Face [repository](https://huggingface.co/datasets/computage/computage_bench), where all **66** DNA methylation datasets from more than **50** studies are assembled and can be retrieved from. All details regarding our methodology of epigenetic aging clocks benchmarking and its results can be found in the paper [...upcoming...].

## Introduction

**DNA methylation** is a chemical modification of DNA molecules that is present in many biological species, including humans. 
Specifically, methylation most often occurs at the cytosine nucleotides in a so-called **CpG context** (cytosine followed by a guanine). 
This modification is engaged in a variety of cellular events, ranging from nutrient starvation responses to X-chromosome inactivation to transgenerational inheritance. 
As it turns out, methylation levels per CpG site change systemically in aging, which can be captured by various machine learning (ML) models called **aging clocks** 
and used to predict an individual’s age. Moreover, it has been hypothesized that the aging clocks not only predict chronological age, but can also estimate 
**biological age**, that is, an overall degree of an individual’s health represented as an increase or decrease of predicted age relative to the general population. 
However, comparing aging clock performance is no trivial task, as there is no gold standard measure of one’s biological age, so using MAE, Pearson’s *r*, or other 
common correlation metrics is not sufficient.

To foster greater advances in the aging clock field, [we developed a methodology and a dataset](https://huggingface.co/datasets/computage/computage_bench) for aging clock benchmarking, ComputAge Bench, which relies on measuring model ability to predict increased ages in samples from patients with *pre-defined* **aging-accelerating conditions** (AACs) relative to samples from 
healthy controls (HC). **We highly recommend consulting the Methods and Discussion sections of our paper before proceeding to use the benchmarking dataset and to build 
any conclusions upon it.**

<p align="center">
<img src="images/fig1.png" alt>

</p>
<p align="center">
<em>ComputAgeBench epigenetic clock construction overview.</em>
</p>

## Usage (benchmarking)

### sklearn-based model

Suppose you have trained your brand new epigenetic aging clock model using the classic `scikit-learn` library. You should save your model as a `pickle` file. Then, the following block of code can be used to benchmark your model. We also implemented imputation of missing values from the R [SeSAMe](https://github.com/zwdzwd/sesame) package and added several published aging clock models for comparison.

```python
from computage import run_benchmark

# first, define a method to impute NaNs for the in_library models
# we recommend using imputation with gold standard values from SeSAMe
imputation = 'sesame_450k'

# for example, take these three clock models for benchmarking
models_config = {
    "in_library":{
        'HorvathV1':{'imputation':imputation},
        'Hannum':{'imputation':imputation},
        'PhenoAgeV2':{'imputation':imputation},
				},
    # here we can define a name of our new model, as well as path
    # to the pickle file (.pkl) that contains it
    "new_models":{
        #'my_new_model_name': {'path':/path/to/model.pkl}
        }
}
# now run the benchmark
bench = run_benchmark(models_config, 
        experiment_prefix='my_model_test',
        output_folder='./benchmark'
        )

# upon completion, the results will be saved in the folder you have specified for output
```

### pytorch-based model
[...upcoming...]

### Explore the dataset
In case you only want to explore our dataset locally, use the following commands to download it:
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='computage/computage_bench', 
    repo_type="dataset",
    local_dir='.')
```

Once downloaded, the dataset can be opened with `pandas` (or any other `parquet` reader).

```python
import pandas as pd

# let's choose a study id, for example, `GSE100264`
df = pd.read_parquet('data/computage_bench_data_GSE100264.parquet').T
# note that we transpose data for a more convenient perception of samples and features

# don't forget to explore metadata (which is common for all datasets):
meta = pd.read_csv('computage_bench_meta.tsv', sep='\t', index_col=0)
```

## Reproducing paper results
All results and plots from the `ComputAgeBench` paper can be reproduced using this [notebook](https://drive.google.com/file/d/1_nrGMUd8oH8ADNWUPNeXHr4ZAJlZOQhm/view?usp=sharing). Alternatively, you can simply clone this repository and run the same `benchmarking.ipynb` notebook locally from the `notebooks` folder.

## Additional information

Aging clock models included in this package.

|     Name     | Year | Number of CpGs | Generation | Extra parameters | Tissues used for training |                                Reference                               |
|:------------:|:----:|:--------------:|:----------:|:----------------:|:-------------------------:|:----------------------------------------------------------------------:|
|    Hannum    | 2013 |       71       |      1     |         —        |           Blood           | [Hannum G. et al.](https://doi.org/10.1016/j.molcel.2012.10.016)       |
|   HorvathV1  | 2013 |       353      |      1     |         —        |        Multi-tissue       | [Horvath S.](https://doi.org/10.1186/gb-2013-14-10-r115)               |
|      Lin     | 2016 |       99       |      1     |         —        |           Blood           | [Lin Q. et al.](https://doi.org/10.18632/aging.100908)                 |
|  VidalBralo  | 2016 |        8       |      1     |         —        |           Blood           | [Vidal-Bralo L. et al.](https://doi.org/10.3389/fgene.2016.00126)      |
|   HorvathV2  | 2018 |       391      |      1     |         —        |        Blood, Skin        | [Horvath S. et al.](https://doi.org/10.18632/aging.101508)             |
|  PhenoAgeV1  | 2018 |       513      |      2     |         —        |           Blood           | [Levine M.E. et al.](https://doi.org/10.18632/aging.101414)            |
|  Zhang19_EN  | 2019 |       514      |      1     |         —        |       Blood, Saliva       | [Zhang Q, et al.](https://doi.org/10.1186/s13073-019-0667-1)           |
|   GrimAgeV1  | 2019 |      1030      |      2     |     Age, Sex     |           Blood           | [Lu A. et al.](https://doi.org/10.18632%2Faging.101684)                |
|   GrimAgeV2  | 2022 |      1030      |      2     |     Age, Sex     |           Blood           | [Lu A. et al.](https://doi.org/10.18632%2Faging.204434)                |
|  PhenoAgeV2  | 2022 |       959      |      2     |         —        |           Blood           | [Higgins-Chen A.T. et al.](https://doi.org/10.1038/s43587-022-00248-2) |
| YingAdaptAge | 2024 |       999      |      1     |         —        |           Blood           | [Ying K. et al.](https://doi.org/10.1038/s43587-023-00557-0)           |
|  YingCausAge | 2024 |       585      |      1     |         —        |           Blood           | [Ying K. et al.](https://doi.org/10.1038/s43587-023-00557-0)           |
|  YingDamAge  | 2024 |      1089      |      1     |         —        |           Blood           | [Ying K. et al.](https://doi.org/10.1038/s43587-023-00557-0)           |

## Cite us
[...coming soon...]

## Contact
For any questions or clarifications, please reach out to: dmitrii.kriukov@skoltech.ru

## Community
Please feel free to leave any questions and suggestions in the issues section. However, if you want a faster and broader discussion, please join our [telegram chat](https://t.me/agingmath). 

## Acknowledgments
We thank the [biolearn](https://bio-learn.github.io/data.html) team for providing an inspiration and a lot of useful tools that were helpful during the initial stages of developing this library. 


