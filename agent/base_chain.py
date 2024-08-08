from langchain.chat_models import ChatOpenAI            # 导入大语言模型
from langchain.prompts import ChatPromptTemplate        # 导入聊天提示模板
from langchain.chains import LLMChain                   # 导入llm链

# 1.实例化大语言模型
llm = ChatOpenAI(temperature=0.9)
# 2. 实例化prompt
prompt = ChatPromptTemplate.from_template(
    "what is the best name to describe a company that makes {product}?")
# 3.实例化llm链
chain = LLMChain(llm=llm, prompt=prompt)      # 现在这个链中，仅需要填入一个{product}即可
# 4.声明product
product = "Queen Size Sheet Set"            # 产品名字
# 5.运行链
response = chain.run(product)
print(response)