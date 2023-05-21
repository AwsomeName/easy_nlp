# 基于trl的PPO实验
# 基于Anthropic/hh-rlhf数据集和bloom560
# Anthropic/hh-rlhf数据集，是Anthropic公司放出的，针对伦理安全的数据集，对于任何大模型都有意义
# bloom560m模型，相比效果比bloom1b更好一点，而训练资源需要更少
# ---- 实验设计 ---------------------
# 整个实验，需要三步：instruct tuning, reward model, PPO
# 第一步，基于hh-rlhf的数据集中，accept部分，训练instruct
# 第二步，基于hh-rlhf的数据集，训练reward model
# 第三步，基于trl库，和第一步的数据，训练PPO


# 下载数据和数据集，模型
download.sh

# instruct tuning， deepspeed
stf.sh

# train reward
train_rm.sh

# PPO 
train_ppo.sh