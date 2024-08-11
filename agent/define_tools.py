# 创建自定义工具，以便可以将agents连接到我们想要的任何内容
from langchain.agents import tool        # 导入工具修饰符,它可以应用于任何函数,并将其转换为链式连接可以使用的工具
from datetime import date
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType
from langchain.agents import load_tools

# 1. 首先使用langchain.agents中的tool工具修饰符,来修饰一个我们自定义的函数
@tool
def time(text: str) -> str:                       # 除了函数的名称外,还应该编写一个详细的文档说明,以便代理知道何时何地使用这个函数
    """返回今天的日期,用于任何与查询今天日期有关的问题.输入应该始终是一个空字符串,
    并且这个函数将始终返回今天的日期---任何日期计算都应该在这个函数之外进行."""
    return str(date.today())

# 2.实例化大语言模型
llm = ChatOpenAI(temperature=0.0)
# 3.声明工具tools
tools = load_tools(['llm-math'], llm=llm)
# 4.使用tools,llm,AgentType来初始化代理,注意在tools上加上我们自定义的工具
agent = initialize_agent(
    tools=tools + [time],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)
# 5.运行代理
result = agent("whats the date today?")
print(result)