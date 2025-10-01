import torch
import argparse
from enum import Enum
import sys
import types

# --- Minimal stubs to satisfy pickle references ---
# This section is necessary to allow torch.load to unpickle the custom
# classes/enums stored in the PyTorch Lightning checkpoint file (.ckpt).
# It creates mock modules and classes so the loader can resolve the paths.

# Root package
sys.modules.setdefault("cancerfoundation", types.ModuleType("cancerfoundation"))


# Submodule: cancerfoundation.loss with LossType enum
class LossType(Enum):
    ORDINALCROSSENTROPY = "ordinal_cross_entropy"
    CORN = "corn"
    MSE = "mse"


loss_mod = types.ModuleType("cancerfoundation.loss")
sys.modules["cancerfoundation.loss"] = loss_mod
setattr(loss_mod, "LossType", LossType)


def extract_and_modify_weights(ckpt_path, output_pth_path):
    """
    Loads a PyTorch Lightning checkpoint, modifies the model's state_dict keys,
    and saves the result to a new .pth file in a single pass.

    Args:
        ckpt_path (str): Path to the input checkpoint file (.ckpt).
        output_pth_path (str): Path to save the modified model weights file (.pth).
    """
    # 1. Load the checkpoint and extract the model weights (state_dict)
    print(f"➡️ Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    original_state_dict = checkpoint["state_dict"]
    print("✅ Checkpoint loaded and state_dict extracted.")

    # 2. Modify the state_dict keys in-memory
    new_state_dict = {}
    print("⚙️  Modifying model state_dict keys...")
    for key, value in original_state_dict.items():
        # First, remove the "model." prefix added by PyTorch Lightning
        new_key = key.replace("model.", "", 1)

        # Apply specific transformations to key names
        if new_key.startswith("encoder") and not new_key.startswith("encoder.layers."):
            # Rename "encoder." to "gene_encoder." for non-layer keys (e.g., cls_token)
            new_key = "gene_encoder" + new_key[len("encoder") :]
        elif "transformer_encoder" in new_key:
            # Standardize "transformer_encoder" to just "encoder"
            new_key = new_key.replace("transformer_encoder", "encoder")

        new_state_dict[new_key] = value
    print("✅ Keys successfully modified.")

    # 3. Save the new, modified state_dict to the final output file
    print(f"💾 Saving modified model to: {output_pth_path}")
    torch.save(new_state_dict, output_pth_path)
    print(f"🎉 Successfully saved modified model to {output_pth_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract, modify, and save model weights from a PyTorch Lightning checkpoint in one pass."
    )
    parser.add_argument("input", help="Path to the input checkpoint file (.ckpt)")
    parser.add_argument(
        "output", help="Path to save the modified model weights file (.pth)"
    )

    args = parser.parse_args()
    extract_and_modify_weights(args.input, args.output)
