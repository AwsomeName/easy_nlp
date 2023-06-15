from huggingface_hub import Repository

repo = Repository(local_dir="/hy-tmp/chat_bloom/")

repo.git_pull()
# repo.push_to_hub(commit_message="update test")
acc_token = "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"

repo.git_add("/hy-tmp/chat_bloom/pytorch_model-00001-of-00004.bin")
repo.git_commit(commit_message="update test_2")

repo.git_push()