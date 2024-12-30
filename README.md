# Profit-driven pre-processing in B2B customer churn modeling using fairness techniques
This repository is associated with the paper "Profit-driven pre-processing in B2B customer churn modeling 
using fairness techniques" by Shimanto Rahman, Bram Janssens & Matthias Bogaert, published in the 
Journal of Business Research. 
The pre-processing techniques—resampling, reweighing, and massaging—are applied to 
an open source customer [churn dataset](https://www.kaggle.com/datasets/ylchang/telco-customer-churn-1113)
and evaluated with the EMPB and AUEPC metrics.

## Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
A full demo of the pre-processing techniques can be found in  
[churn_preprocessing.ipynb](https://github.com/ShimantoRahman/churn-preprocessing/blob/main/churn_preprocessing.ipynb).

This is a dummy use-case of the pre-processing techniques:
```python
import numpy as np
from empulse.metrics import auepc_score
from empulse.models import BiasRelabelingClassifier, BiasResamplingClassifier, BiasReweighingClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
X, y = make_classification()
clv = np.random.rand(X.shape[0]) * 100
high_clv = clv > np.quantile(clv, 0.8)

resampling_model = BiasResamplingClassifier(estimator=LogisticRegression())
reweighing_model = BiasReweighingClassifier(estimator=LogisticRegression())
massaging_model = BiasRelabelingClassifier(estimator=LogisticRegression())

resampling_model.fit(X, y, protected_attr=high_clv)
reweighing_model.fit(X, y, protected_attr=high_clv)
massaging_model.fit(X, y, protected_attr=high_clv)

resampling_y_pred = resampling_model.predict_proba(X)[:, 1]
reweighing_y_pred = reweighing_model.predict_proba(X)[:, 1]
massaging_y_pred = massaging_model.predict_proba(X)[:, 1]

print(f'AUEPC Resampling: {auepc_score(y, resampling_y_pred, clv=clv)}')
print(f'AUEPC Reweighing: {auepc_score(y, reweighing_y_pred, clv=clv)}')
print(f'AUEPC Massaging: {auepc_score(y, massaging_y_pred, clv=clv)}')
```

Comprehensive documentation of the functions can be found in the 
[Empulse](https://shimantorahman.github.io/empulse/index.html) package.

## Reference
- Rahman, S., Janssens, B., & Bogaert, M. (2025). 
Profit-driven pre-processing in B2B customer churn modeling using fairness techniques. 
Journal of Business Research, 189, 115159.