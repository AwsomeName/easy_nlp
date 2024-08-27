from openai import OpenAI
import os
import csv
import json

# 初始化OpenAI客户端
client = OpenAI(
    #api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    api_key="sk-4c161f6cda4a40aabda543ebe7cb1ebb",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

tools = [
    # 工具 获取虚拟地点
    {
        "type": "function",
        "function": {
            "name": "extract_navigation_location",
            "description": "当提到地点的时候，调用此工具",
            "parameters": {}  
        }
    }
]
# 模拟地点查询工具
def extract_navigation_location(question):
    prompt=f"""
        提取问题中包含的地点名称，不要做任何修改，直接输出。输出格式为json
        例如：
        输入：去北京怎么走？
        输出：{"location": "北京"}
                    
    """
            
    messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': question}
            ]
    
    res = get_response(messages)
    
    return res

# 封装模型响应函数
def get_response(messages):
    completion = client.chat.completions.create(
        model="qwen-max",
        messages=messages
        )
    return completion.model_dump()

# 封装模型响应函数
def get_response_with_tools(messages):
    completion = client.chat.completions.create(
        model="qwen-max",
        messages=messages,
        tools=tools
        )
    return completion.model_dump()

def call_with_messages(query):
    print('\n')
    messages = [
        {'role': 'system', 'content': "你是一个有用的助手，当用户提到地点，那么使用工具"},
        {"content": input(query), "role": "user"}
    ]
    print("-"*60)
    # 模型的第一轮调用
    i = 1
    first_response = get_response_with_tools(messages)
    assistant_output = first_response['choices'][0]['message']
    print(f"\n第{i}轮大模型输出信息：{first_response}\n")
    if  assistant_output['content'] is None:
        assistant_output['content'] = ""
    messages.append(assistant_output)
    # 如果不需要调用工具，则直接返回最终答案
    if assistant_output['tool_calls'] == None:  # 如果模型判断无需调用工具，则将assistant的回复直接打印出来，无需进行模型的第二轮调用
        print(f"无需调用工具，我可以直接回复：{assistant_output['content']}")
        return
    # 如果需要调用工具，则进行模型的多轮调用，直到模型判断无需调用工具
    while assistant_output['tool_calls'] != None:
        # 如果判断需要调用查询天气工具，则运行查询天气工具
        if assistant_output['tool_calls'][0]['function']['name'] == 'extract_navigation_location':
            tool_info = {"name": "extract_navigation_location", "role":"tool"}
            # 提取位置参数信息
            # location = json.loads(assistant_output['tool_calls'][0]['function']['arguments'])['properties']['location']
            tool_info['content'] = extract_navigation_location(query)
       
        print(f"工具输出信息：{tool_info['content']}\n")
        print("-"*60)
        break


if __name__ == '__main__':
    # call_with_messages("今天天气怎么样")
    call_with_messages("导航去北京")