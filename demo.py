"""
Quickstart Demo for S2D FusionNet.

This script demonstrates how to instantiate S2D FusionNet, feed dummy HRRP
radar profiles through the network, and obtain classification logits and
global feature embeddings.
"""

import sys
import os
import torch

# Ensure current folder is in Python search path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from framework import S2DFusionNet


def main():
    print("=" * 65)
    print("      S2D FusionNet - Minimal Quickstart & Verification Demo      ")
    print("=" * 65)

    # 1. Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1] Target Device: {device}")
    if device.type == "cuda":
        print(f"    GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. Instantiate S2DFusionNet with standard paper configuration
    num_classes = 10
    dim = 128
    k = 12
    heads = 4
    petype = "rope"
    method = "RPM"

    print("\n[2] Instantiating S2DFusionNet:")
    print(f"    - 2D Representation:       {method}")
    print(f"    - Dynamic KNN Neighbors:   K = {k}")
    print(f"    - Feature Dimension:       dim = {dim}")
    print(f"    - Attention Heads:         heads = {heads}")
    print(f"    - Positional Prior:        {petype.upper()} (Rotary Position Embedding)")
    print(f"    - Number of Target Classes:{num_classes}")

    model = S2DFusionNet(
        method=method,
        k=k,
        heads=heads,
        num_classes=num_classes,
        dim=dim,
        gcn_layers=1,
        tf_layers=1,
        petype=petype,
    ).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    - Trainable Parameters:    {total_params:,} ({total_params / 1e3:.2f} K)")

    # 3. Create dummy input batch
    batch_size = 8
    seq_len = 64  # Downsampled sequence length
    dummy_input = torch.randn(batch_size, 1, seq_len, device=device)
    print(f"\n[3] Input HRRP Batch Shape:    {tuple(dummy_input.shape)} (B, C, L)")

    # 4. Forward pass
    with torch.no_grad():
        logits, global_features = model(dummy_input, return_feature=True)

    print("\n[4] Forward Pass Outputs:")
    print(f"    - Output Logits Shape:     {tuple(logits.shape)} (B, num_classes)")
    print(f"    - Global Features Shape:   {tuple(global_features.shape)} (B, dim)")
    print(f"    - Sample Predicted Classes:{logits.argmax(dim=-1).cpu().tolist()}")

    # 5. Test all available S2D mapping strategies
    print("\n[5] Testing all S2D Mapping Strategies:")
    for test_method in ["RPM", "GASF", "GADF", "Hybrid"]:
        test_model = S2DFusionNet(
            method=test_method,
            k=k,
            heads=heads,
            num_classes=num_classes,
            dim=dim,
        ).to(device)
        test_model.eval()
        with torch.no_grad():
            test_out = test_model(dummy_input)
        print(f"    - Method: {test_method:<7} -> Output Logits: {tuple(test_out.shape)} [OK]")

    print("\n" + "=" * 65)
    print(" Verification successful! S2D FusionNet is ready for deployment. ")
    print("=" * 65)


if __name__ == "__main__":
    main()
