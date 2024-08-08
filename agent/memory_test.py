from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage

api_key = "evllm"
api_url = "http://localhost:8000/v1"
modal= "chatglm"
llm = ChatOpenAI(model_name=modal,openai_api_key=api_key,openai_api_base=api_url)

# # 1.实例化大语言模型
# llm = ChatOpenAI(temperature=0.7)
# 2.设置语言模型的SystemMessage
llm([SystemMessage(content='你是一个精通同态加密的教授，你会解答学生关于同态加密的问题。')])
# 3.实例化会话标记缓存记忆对象，参数llm=大语言模型，max_token_limit=最大标记限制
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=60)
# 4.实例化会话链
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
# 5.模拟连续对话
memory.save_context({"input": "AI is what?!"}, {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"}, {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"},  {"output": "Charming!"})  
# 6.会话记忆中会根据max_token_limit来缓存
print(memory.load_memory_variables({}))