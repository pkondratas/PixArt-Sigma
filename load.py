from datasets import load_dataset

ds = load_dataset(
    "CSU-JPG/TextAtlas5M", 
    "TextVisionBlend", 
    split="train",
    cache_dir="./huggingface"
)