from datasets import load_dataset, load_from_disk


# ds = load_dataset("HuggingFaceH4/oasst1_en")

# ds.save_to_disk("./oasst1_en/")


ds = load_from_disk("./oasst1_en")