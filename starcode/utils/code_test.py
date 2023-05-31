from datasets import load_dataset, load_from_disk
test_data = load_dataset("openai_humaneval")
test_data.save_to_disk('/root/data/humaneval')

# test_data = load_from_disk('/root/data/humaneval')

# DatasetDict({
#     test: Dataset({
#         features: ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point'],
#         num_rows: 164
#     })
# })

print(test_data)