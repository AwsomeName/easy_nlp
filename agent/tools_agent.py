from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from ChatTools import ServerRecommendTool, PetConsultationTool, GenerateNameTool, BehaviorInterpretationTool
from langchain.agents import AgentType

# 1.导入记忆机制
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
# 2.导入大语言模型
llm = ChatOpenAI(temperature=0.0)
# 3.加载工具列表
tools = [ServerRecommendTool(), PetConsultationTool(), GenerateNameTool(), BehaviorInterpretationTool()]
# 4.创建agent
agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,            # 不同的agent类型
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,                      # 查找答案的最高次数
    early_stopping_method='generate',
    memory=memory
)
# 5.agent的使用
message = input(">>")
try:
    response = agent.run(message)
except ValueError as e:
    response = '不好意思，请询问我xxxx相关的问题'
print(response)