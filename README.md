# Hierarchical Optimal Transport for Unsupervised Domain Adaptation - Submitted to Machine Learning Journal

## Installation and Dependencies

**HOT-DA** is based on `NumPy`, `Pandas`, `Scikit-Learn` and `POT`.
So, make sure these packages are installed. For example, you can install them with `pip`:

```
pip3 install numpy pandas POT
```

It is recommend to use `Python 3.8.8` from [Anaconda](https://www.anaconda.com/) distribution. All the codes for the article are tested on macOS Big Sur Version 11.6


## Scripts for experiments:
To reproduce the results, the following `jupyter notebooks` are provided:

 1 - `/Moons/Moons.ipynb`

 2 - `/Digits/Mnist_USPS.ipynb` and `/Digits/USPS_Mnist.ipynb`

 3 - `/OfficeCaltech/OfficeCaltech.ipynb`

 4 - `/OfficeHome/OfficeHome.ipynb`


 ## Scripts for figures:

● Wasserstein-Spectral clustering - `/Wasserstein-Spectral_Clustering.ipynb`

● Decision boundary on moons dataset - `/Moons_DecisionBoundary.ipynb`

● Structure imbalance sensitivity analysis - `/Structure_imbalance_sensitivity_analysis.ipynb`


## Other scripts:

● Main class for our algorithm - `/HOTDA.py`


## Datasets:

● We release `Moons` and `Digits` `datasets` (within each of the two files: Moons and Digits)

● `OfficeHome` and `Office-Caltech` `datasets` are too voluminous to be uploaded on Github (some files are larger than GitHub's recommended maximum file size of 50.00 MB). Please download them from the following links:

`OfficeHome`: https://mega.nz/folder/R1lUVYgQ#1_LvnpvixZmFsZNmztzrvg

`Office-Caltech`: https://mega.nz/folder/A983DYiB#j3po3gRqIuycsJAGn5vBww
