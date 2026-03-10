#!/bin/bash
# Run locally:
#   devcontainer exec --workspace-folder /local/home/rvander/CancerFoundation python check_bionemo.py
#
# Run on cluster:
#   srun --container-image=... python check_bionemo.py

python check_bionemo.py
