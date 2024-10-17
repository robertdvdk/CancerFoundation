# CancerGPT: A single-cell RNA sequencing foundation model to decipher drug resistance in cancer

[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org) &nbsp;
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/BoevaLab/CancerGPT/blob/main/LICENSE)

We present **CancerGPT**, a novel single-cell RNA-seq foundation model (scFM) trained exclusively on malignant cells. Despite being trained on only one million total cells, a fraction of the data used by existing models, CancerGPT outperforms other scFMs in key tasks such as zero-shot batch integration and drug response prediction. During training, we employ tissue and technology-aware oversampling and domain-invariant training to enhance performance on underrepresented cancer types and sequencing technologies. We propose survival prediction as a new downstream task to evaluate the generalizability of single-cell foundation models to bulk RNA data and their applicability to patient stratification. CancerGPT demonstrates superior batch integration performance and shows significant improvements in predicting drug responses for both unseen cell lines and drugs. These results highlight the potential of focused, smaller foundation models in advancing drug discovery and our understanding of cancer biology.

## Installation

### Prerequisites

Make sure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your machine.

### Step-by-Step Guide

1. **Clone the repository**:

   ```bash
   git clone https://github.com/BoevaLab/CancerGPT.git
   cd CancerGPT
   ```
2. **Create the Conda environment**:
   ```bash
   conda env create -f environment.yml
   ```
3. **Activate environment**:
   ```bash
   conda activate cancergpt
   ```
4. **Download pretrained model weights**:

   Please download the pretrained model from [this link](https://polybox.ethz.ch/index.php/s/pZR9VH7uEHwO5CL), unzip it, and place it in the following directory: ```model/assets```.

## Generate embeddings
Please consult ```tutorial/embeddings_tutorial.ipynb``` for a tutorial on how to generate embeddings with CancerGPT for your scRNA-seq data.
