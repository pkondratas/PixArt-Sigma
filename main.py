# from datasets import load_dataset

# ds = load_dataset("CSU-JPG/TextAtlas5M", "CleanTextSynth", split="train[:1%]")

# from datasets import load_dataset

# ds = load_dataset("PixArt-alpha/pixart-sigma-toy-dataset")

import pandas as pd

df = pd.read_parquet("C:\\Users\\Paul James\\.cache\\huggingface\\hub\\datasets--CSU-JPG--TextAtlas5M\\snapshots\\945d5e3053619ea80eb8e616d6cdddd85e7aa4ef\\StyledTextSynth\\train-00000-of-00290.parquet")

print(df.head())

# breakpoint()

# python train_scripts/train.py configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms.py --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth --work-dir output/first --debug
