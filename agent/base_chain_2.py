from langchain.chains import SimpleSequentialChain          # 导入简单顺序链
from langchain.prompts import ChatPromptTemplate            # 导入聊天提示模板
from langchain.chat_models import ChatOpenAI                # 导入大语言模型
from langchain.chains import LLMChain                   # 导入llm链
# 1.实例化大语言模型
llm = ChatOpenAI(temperature=0.9)
# 2.声明两个llm链，都是有一个输入和一个输出，其中llm链1的输出正好是链2的输入
first_prompt = ChatPromptTemplate.from_template(
    "对于一家生产{product}的公司，为其取一个最合适的名字")
chain_one = LLMChain(llm=llm, prompt=first_prompt)
second_prompt = ChatPromptTemplate.from_template("为{company_name}公司写一篇50字的描述")
chain_two = LLMChain(llm=llm, prompt=second_prompt)
# 3.声明简单顺序链
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)
# 4.设置product
product = "Queen Size Sheet Set"            # 产品名字
# 4.运行链
response = overall_simple_chain.run(product)
print(response)