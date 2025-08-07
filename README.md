## Installation

### Prerequisites

1. **Docker**
2. **NVIDIA GPU Drivers**
3. **NVIDIA Container Toolkit**

   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
4. **Devcontainer**

   **VSCode (Recommended)**:
   
   https://code.visualstudio.com/docs/devcontainers/containers

   **CLI**:

   https://code.visualstudio.com/docs/devcontainers/devcontainer-cli
5. **Weights and Biases (optional)**


### Step-by-Step Guide
1. **Clone the repository**:
   ```bash
   git clone https://github.com/BoevaLab/CancerFoundation.git
   cd CancerFoundation
   ```

2. **Launch the dev container**:

   **VSCode**:
      Open the cloned repo folder in VSCode. A notification will open prompting to "Reopen in Container". Click it. VSCode will build the image and start the container. 
   
   **CLI**:
      In the command line, build the devcontainer and then open a terminal inside it as follows:
      ```bash
      devcontainer up --workspace-folder .
      devcontainer exec --workspace-folder . bash
      ```

3. **Run the pretraining script**:
   Now, either in VSCode or in the CLI, call:
   ```bash
   bash pretrain.sh
   ```
   Optionally, call the training script inside a tmux session.


### Logging with Weights & Biases
   If you want to use Weights & Biases to track your training run, put the following inside .devcontainer/devcontainer.env:
   ```
   WANDB_API_KEY={YOUR_API_KEY}
   ```
   You can find this key [here](https://wandb.ai/authorize).