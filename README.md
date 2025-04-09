## Installation

### Prerequisites

Make sure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your machine.

### Step-by-Step Guide

1. **Clone the repository**:

   ```bash
   git clone https://github.com/BoevaLab/CancerFoundation.git
   cd CancerFoundation
   ```
2. **Create the Conda environment**:
   ```bash
   conda env create -f environment.yml
   ```
3. **Activate environment**:
   ```bash
   conda activate cancerfoundation
   ```
4. **Download pretrained model weights**:

   Please download an exemplary training dataset from [this link](https://polybox.ethz.ch/index.php/s/ANH0oCX4Mkw4Nos), unzip it, and place it in this directory.

## Training

To start an exemplary training run as a SLURM job, call:
```bash
bash pretrain.sh
```

