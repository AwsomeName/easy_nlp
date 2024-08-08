# 用于llm-math和wikipedia的代理
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
# 1.实例化一个大语言模型
api_key = "evllm"
api_url = "http://localhost:8000/v1"
modal= "chatglm"
llm = ChatOpenAI(model_name=modal,openai_api_key=api_key,openai_api_base=api_url)
# llm = ChatOpenAI(temperature=0.0)
# 2.选择语言模型要接入的工具
tools = load_tools(["llm-math", "wikipedia"], llm=llm)
# 3.使用tools，llm，代理类型来初始化agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)
# 4.使用代理来解决我们提出的数学问题
result = agent("计算99的5次方模12的结果.")
print(result)
# 5.使用代理来解决我们提出的关于维基百科上的问题
question = """Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"""
result = agent(question)
print(result)