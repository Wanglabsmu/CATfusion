import argparse
import torch

import os
import sys
os.chdir(sys.path[0])
sys.path.append(
    "/home/huyongfei/PycharmProjects/VLP_wsi_RNAseq/wsi_omics_fusion/CATfusion_survival_prediction")
from models.modeling_CATfusion import CATfusion, CONFIGS  # NOQA: E402

# "rna_seq", "mirna_seq", "copynumbervariation", "dnamethylationvariation", "snp"
config = CONFIGS["CATfusion"]
print("config", config)
model = CATfusion(config, zero_head=True, num_classes=1)

imgs = torch.randn(2, 100, 768)
omics_input_dict = {
    "rna_seq": torch.randn(2, 10, 1024),
    "mirna_seq": torch.randn(2, 2, 1024),
    "copynumbervariation": torch.randn(2, 27, 1024),
    "dnamethylationvariation": torch.randn(2, 8, 1024),
    "snp": torch.randn(2, 10, 1024),
}

loss_cox, logits, x = model(imgs, omics_input_dict, survtime=torch.tensor(
    [1123, 968]).float(), censor=torch.tensor([1.0, 1.0]).float())
print("loss", loss_cox)
print("logits", logits.shape)
print("transformer output size", x.shape)

# loss tensor([0.2320], grad_fn=<ReshapeAliasBackward0>)
# logits torch.Size([2, 1])
# transformer output size torch.Size([2, 158, 768])
