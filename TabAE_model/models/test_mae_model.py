import torch
import torch.nn as nn
import os
import sys
os.chdir(sys.path[0])
sys.path.append(os.getcwd())

from mae_model import MaskedAutoencoderViT  # NOQA: E402


model = MaskedAutoencoderViT(sequence_length=10,
                             embed_dim=1024, depth=4, num_heads=16,
                             decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
                             mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)

inputs = torch.randn(3, 10, 1024)
loss, pred, mask = model(inputs)
print("loss", loss.shape)
print("pred", pred.shape)
print("mask", mask.shape)
# loss torch.Size([])
# pred torch.Size([3, 10, 1024])
# mask torch.Size([3, 10])

###############################################################################
# predict
model.eval()

latent, mask, ids_restore = model.forward_encoder(
    inputs, mask_ratio=0.0)
print("latent", latent.shape)
# latent torch.Size([3, 11, 1024])



# valid_output_embedding.append(latent[:, 0, :].cpu().detach().numpy())
# valid_output_embedding.append(torch.cat([latent[:, 0, :], torch.mean(latent[:, 1:, :], dim=1)], dim=1).cpu().detach().numpy())
