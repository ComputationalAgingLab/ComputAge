# **Multi-model platform for predicting biological age using DNA methylation and other age-associated data**
![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![made with love](https://img.shields.io/badge/made%20with%20%E2%9D%A4%EF%B8%8F-8A2BE2)

A repository containing the code accompanying the thesis "Multi-model platform for predicting biological age using DNA methylation and other age-associated data" by Aiusheeva A., Khairetdinova A., Kriukov D., Efimov E. ([link to thesis](https://docs.google.com/document/d/14n0dbZ__1WixYWfHOoA_MzKpy7_U_4bF0A2hsUm4ggA/edit?usp=sharing), [Link to slides](https://docs.google.com/presentation/d/1mGrtdA-2_gAEoWWa5XMw-bVd3PpfNBnS7KbofaOf0Gs3/edit?usp=sharing))

## Description

DNA methylation (DNAm) has been widely used to estimate epigenetic age as a proxy metric of biological age in various research settings, for example, to test if some pro-longevity intervention such as rapamycin treatment or caloric restriction affects aging in short-term experiments. Multiple DNAm-based (and other omics-based) clock models have been described, but all of them perform differently from each other, so researchers often resort to several clocks simultaneously to substantiate their findings. Unfortunately, since all of these clocks were developed independently, they must be installed from separate places, and then processed and trained anew, which is highly inconvenient and might affect reproducibility. Currently, there is an R (Bioconductor) package called methylclock which allows to generate age predictions using a number of existing clock models, to check their correlation metrics, and to visualize them. However, it has several significant drawbacks (e.g., compatibility errors; lack of dataset normalization, QC, batch effect correction, and other processing steps; and lack of some well-known and widely employed clocks), as well as it`s focused on DNAm only, which all makes this package of limited use. 

We offer convenient and comprehensive tool for biological age estimation and comparison. This tool is the Python module Computage for fast and easy-to-use estimation of biological age.



### Content of Repository

The repository includes 7 jupyter notebooks named by the type of data under analysis with our analytical framework. The first three notebooks contain main results of the paper.

- `workbook_dataset_assembly.ipynb` - Data assembling for the study
- `lin_models_estimation_full.ipynb` - Linear models estimation workbook
- `imputation_linmodels.ipynb` - Testing `average`, `none`, `sesame450k` imputation in linear models
- `imputation_hc_phenoage.ipynb` - Testing `average`, `none`, `sesame450k` imputation in `phenoage2018` (light version of the notebook above)
- `requirements.txt` - python modules that were used 


# **Installation**

```bash
git clone https://github.com/ComputationalAgingLab/ComputAge
cd ComputAge
git checkout dev_clocks
```

## **Requirements**
```bash
conda env create --name computage --file dev_clocks.yml
conda activate computage
```
*or (preferable)* 
```bash
python3 -m venv .computage_venv
source .computage_venv/bin/activate
pip install -r requirements.txt
python3 -m ipykernel install --user --name=ComputAge
```
***

# **Usage tutorial**

## Download modules and data

```python
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, median_absolute_error
from computage.utils.data_utils import download_meta, download_dataset
from computage.models_library.model import LinearMethylationModel

meta = download_meta('./meta_table_datasets.xlsx')
download_dataset(meta, 'GSE132203', '.')
df = pd.read_pickle('GSE132203.pkl')
```
## Example with `phenoage` model, imputation by `average`

Case with GSEID `GSE132203` from GEO database
```python
X = pd.DataFrame(df['data'])
meta = pd.DataFrame(df['meta'])
y = pd.DataFrame(meta['Age'])
y_test = y.rename(columns={'Age': 'age'})


model_phenoage = LinearMethylationModel(name='phenoage2018', imputation='average')
y_pred_test = model_phenoage.predict(X)
       
print(median_absolute_error(y_test, y_pred_test))
print(r2_score(y_test, y_pred_test))

```

## [Usage notebook for user](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/Example.ipynb)


## __List of available ready-to-use models:__
- ***Linear blood models*** : [`hrsinchphenoage`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/HRSInCHPhenoAge.csv),
 [`lin2016blood_99cpgs`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/Lin2016Blood_99CpGs.csv),
 [`yingdamage`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/YingDamAge.csv),
 [`yingadaptage`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/YingAdaptAge.csv),
 [`hannum2013blood`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/Hannum2013Blood.csv),
 [`dec2023encen100`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/Dec2023ENCen100.csv),
 [`yingcausage`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/YingCausAge.csv),
 [`horvath2018`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/Horvath2018.csv),
 [`vidal-bralo2016blood`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/Vidal-Bralo2016Blood.csv),
 [`lin2016blood_3cpgs`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/Lin2016Blood_3CpGs.csv),
 [`zhangblup2019`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/ZhangBLUP2019.csv),
 [`zhangenclock2019`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/ZhangENClock2019.csv),
 [`phenoage2018`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/PhenoAge2018.csv),
 [`dec2023encen40`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/Dec2023ENCen40.csv), [`horvath2018`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/Horvath2018.csv), [`horvath2013_shrunken`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/Horvath2013_Shrunken.csv), [`horvath2013`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/Horvath2013.csv), [`han2020blood`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/Han2020Blood.csv)

More info [`computage/models_library/ModelsDescription(upd20240805).csv`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/ModelsDescription(upd20240805).csv)

# **Data and models availability**
## Datasets Used in Study
All datasets used in the study can be found in GEO vi IDs: `GSE69138`, `GSE59685`, `GSE203399`, `GSE32148`, `GSE87640`, `GSE42861`, `GSE62867`, `GSE56581`, `GSE107143`, `GSE62003`, `GSE53840`, `GSE87648`, `GSE49909`, `GSE56046`. More info about collected data in [`meta_table_datasets.xlsx`](https://docs.yandex.ru/docs/view?url=ya-disk-public%3A%2F%2F7ywsWVjy4DeeWAZKkSJjw8scu7IAQL3ZWJt8jlz%2FSR%2BLxm%2Fe%2FAss5aQ9fRfwVXI%2Bq%2FJ6bpmRyOJonT3VoXnDag%3D%3D&name=meta_table_datasets.xlsx&nosw=1)

For imputation by `sesame450k` [`computage/models_library/raw_models/sesame_450k_median.csv`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/raw_models/sesame_450k_median.csv) data was used.

## Models Used in Study
All linear models used in the study can be found in [`computage/models_library/ModelsDescription(upd20240805).csv`](https://github.com/ComputationalAgingLab/ComputAge/blob/dev_clocks/computage/models_library/ModelsDescription(upd20240805).csv)

## Contact
For any questions or clarifications, please reach out to: aryuna.ayusheeva.1998@gmail.com, khairetdinova.studies@gmail.com, dmitrii.kriukov@skoltech.ru