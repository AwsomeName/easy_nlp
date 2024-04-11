from datasets import load_dataset, DatasetDict, load_from_disk
from datasets.utils.info_utils import VerificationMode

common_voice = DatasetDict()
# data_path = "/home/lc/data_a/mozilla-foundation/common_voice_11_0"
# data_path = "/Users/lc/code/easy_nlp/data/train_100.txt"
data_path = "/Users/lc/code/easy_nlp/data/"

# local_ds = load_dataset(
    # data_path, 
    # name="hi", 
    # data_dir=data_path, 
    # verification_mode=VerificationMode.NO_CHECKS)
# print(local_ds)

data_files = {'train': 'train_100.txt', 'test': 'test_10.txt'}
ds = load_dataset(data_path, data_files=data_files)

print(ds)
print(ds['test'])
print(ds['test'][0])