# python-agent
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
# 1.实例化一个大语言模型
llm = ChatOpenAI(temperature=0.0)
# 2.创建一个python代理,传入大语言模型,PythonREPL工具
agent = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True
)
# 3.设置想要解决的问题,这里是想要实现一个对人名排序的问题,并生成该message
customer_list = [["Harrison", "Chase"],
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"],
                 ["Geoff", "Fusion"],
                 ["Trance", "Former"],
                 ["Jen", "Ayai"]]
template = """按姓氏和名字对这些客户排序,然后打印输出:{customer_list}"""
prompt = ChatPromptTemplate.from_template(template)
message = prompt.format_messages(customer_list=customer_list)
# 4.将message传给代理,得到代理的回答
result = agent.run(message)
print(result)