#!/bin/bash


# git lfs install
# git clone https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers
# python tools/download.py 

# ^^^ Reikia virsutiniu pries leidziant sitas ^^^

python ./load.py
python ./setup.py

# python train_scripts/train.py configs/pixart_sigma_config/MOD_PixArt_sigma_xl2_img512_internalms.py --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth --work-dir output/first --debug