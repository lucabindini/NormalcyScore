# NormalcyScore
_Dealing with Uncertainty in Contextual Anomaly Detection_ [TMLR 2026]

Implementation of the **normalcy score (NS)** framework presented in *“Dealing with Uncertainty in Contextual Anomaly Detection”*.  
NS is a **contextual anomaly detection** method that models both **aleatoric** and **epistemic** uncertainty via **heteroscedastic Gaussian process regression (HGPR)**, returning:
- an anomaly score (expected NS)
- an uncertainty estimate through a **95% Highest Density Interval (HDI)** on the score

---

## Libraries Used

- `numpy`, `pandas`, `scipy`
- `scikit-learn`
- `tensorflow`, `gpflow`
- `arviz` (HDI computation)
- `pyod` (optional baselines / utilities)
- `statsmodels`, `patsy` (optional utilities)
- `rich` (console logging)

---

## Repository Structure


- `datasets/`  
  Put your **CSV datasets** here.

- `NS.py`  
  Implements the **normalcy score**: score definition and computation (and uncertainty-related outputs).

- `test_NS.py`  
   Runnable script to test NS on a dataset from `datasets/` and save results.

- `ContextualAnomalyInject.py`  
  Utility to **inject contextual anomalies** and generate `ground_truth`.


The code structure and the anomaly injection script were implemented following the QCAD [DAMI 2023] implementation available [here](https://github.com/ZhongLIFR/QCAD).



## Citing the Paper

If you use this code, please cite:

```bibtex
@article{
bindini2026dealing,
title={Dealing with Uncertainty in Contextual Anomaly Detection},
author={Luca Bindini and Lorenzo Perini and Stefano Nistri and Jesse Davis and Paolo Frasconi},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=yLoXQDNwwa},
note={}
}
```

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />All material is available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** you've made.