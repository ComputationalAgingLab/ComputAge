# **Installation**

```bash
git clone https://github.com/ComputationalAgingLabComputAge
```
```bash
conda env create --name computage --file dev_clocks.yml

conda activate computage
```
*or* 
```bash
pip3 install requirements.txt
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



[Usage notebook](link to notebook)


## __List of available models:__
- ***Linear*** : `hrsinchphenoage`,
 `lin2016blood_99cpgs`,
 `yingdamage`,
 `yingadaptage`,
 `hannum2013blood`,
 `dec2023encen100`,
 `yingcausage`,
 `horvath2018`,
 `vidal-bralo2016blood`,
 `lin2016blood_3cpgs`,
 `zhangblup2019`,
 `zhangenclock2019`,
 `phenoage2018`,
 `dec2023encen40`, `horvath2018`, `horvath2013_shrunken`, `horvath2013`, `han2020blood`
- ***Nonlinear*** : `epitoc2`

More info [link table name](link to csv table)