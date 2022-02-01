# PORTIA

Source code for the paper "Fast and Accurate Inference of Gene Regulatory Networks through Robust Precision Matrix Estimation", by Passemiers et al. PORTIA builds on power transforms and covariance matrix inversion to approximate gene regulatory networks (GRNs), and is orders of magnitude faster than other existing tools (as of August 2021).

---

In this repository, you will learn how to reproduce the results presented in our manuscript. All inferred GRNs and intermadiate computations are present on the repository as well.

### PORTIA

Our GRN inference tool is available at: https://github.com/AntoinePassemiers/PORTIA

---

### Data availability

#### DREAM challenges

For accessing the datasets from the DREAM challenges, you will have to create an account on Synapse:
https://python-docs.synapse.org/build/html/index.html#accessing-data

Optionally, you can create a `SYNAPSE_USERNAME` environment variable:
```bash
SYNAPSE_USERNAME=my_username_or_email
export SYNAPSE_USERNAME
```

For your information only, these are the competitions links:
https://www.synapse.org/#!Synapse:syn2853594/wiki/71567 (DREAM3)
https://www.synapse.org/#!Synapse:syn3049712/wiki/74628 (DREAM4)
https://www.synapse.org/#!Synapse:syn2787209/wiki/70351 (DREAM5)

#### MERLIN-P

Download the MERLIN-P data repository located at: https://github.com/Roy-lab/merlin-p_inferred_networks
Uncompress and place the repository folder in `PORTIA-MANUSCRIPT/data` under the name `merlin-p_inferred_networks-master`.

### Repository organisation

- All figures, including some present in the paper, are available in `/figures`.
- Generated tables are available in `/tables`.
- Inferred networks of all methods are provided for each dataset in `/inferred-networks`.
- The source code for performing the evaluations is present in `/evalportia`.
- Intermediate data (in binary format) for computing p-values (MERLIN-P datasets only) is located in `/evalportia`.

### Install dependencies and GRN inference tools

#### Python dependencies

```bash
pip3 install -r requirements.txt
```

#### R dependencies, GENIE3 and TIGRESS

```r
# R dependencies
install.packages(c("devtools", "foreach", "plyr", "doRNG", "glmnet", "randomForest"))

# GENIE3
# http://bioconductor.org/packages/release/bioc/html/GENIE3.html
if (!requireNamespace("BiocManager", quietly=TRUE))
    install.packages("BiocManager")
BiocManager::install("GENIE3")

# TIGRESS
# https://github.com/jpvert/tigress
library(devtools)
install_github("jpvert/tigress")
```

#### ENNET

Download and uncompress the ENNET repository from https://github.com/slawekj/ennet
and build it from the source:

```bash
R CMD build ennet && R CMD INSTALL ennet
```

#### NIMEFI

Download NIMEFI at:
http://bioinformatics.intec.ugent.be/nimefi/nimefi/index.html

Uncompress and save the location of the "NIMEFI" folder as an environment variable:
```bash
NIMEFI_LOCATION=/path/to/NIMEFI
export NIMEFI_LOCATION
```

#### ARACNe-AP

Follow the installation instructions at:
https://github.com/califano-lab/ARACNe-AP

Save the path to the generated .jar file as an environment variable:
```bash
ARACNE_AP_LOCATION=/path/to/ARACNe-AP/dist/aracne.jar
export ARACNE_AP_LOCATION
```

#### PLSNET

Download the Matlab source code on the publication page:
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-1398-6#Sec17

Uncompress and save the location of the "PLSNET" folder as an environment variable:
```bash
PLSNET_LOCATION=/path/to/PLSNET
export PLSNET_LOCATION
```

---

### Run scripts

All the necessary scripts are located at the root of this repository.

#### Infer GRNs

To infer GRNs from the DREAM3 datasets with PORTIA for example, please use the following command:
```bash
python3 infer-dream3.py portia
```
To remove knock-out data from the datasets, you can add the `-noko` optional argument (valid for `dream3`, `dream4` and `dream5` only).

#### Evaluate GRNs

To evaluate generated GRNs for the `merlin-p` datasets for example, please use the following command:
```bash
python3 evaluate-merlin-p.py
```

This will automatically save figures and tables.

---
