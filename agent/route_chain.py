from langchain.chains.router import MultiPromptChain       # 导入多提示链
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser   # 导入LLM路由链，路由输出解释器
from langchain.prompts import PromptTemplate               # 导入提示模板
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# 1.声明多个template
physics_template = """你是一个聪明的物理学教授\
你很擅长用简单易懂的语言回答有关物理的问题\
当你不知道问题的答案时，你承认你不知道。
这里是问题{input}
"""
math_template = """你是一个很好的数学家。\
你很擅长回答数学问题。\
你是如此的优秀，因为你能够把难题分解成各个组成部分，并回答各个组成部分，\
然后把他们组合在一起，从而回答更广泛的问题。
这里是问题{input}
"""
history_template = """你是一个非常好的历史学家。\
你对各个时期的人物、事件以及背景都有很深入的了解。\
你有思考、反思、辩论、讨论和评价过去的能力。\
你尊重历史证据，以及利用这些证据来支持你的解释和判断能力。
这里是问题{input}
"""
computerscience_template = """你是一个成功的计算机科学家。\
你有创造力，协作精神，前瞻性思维，自信，有很强的解决问题的能力，\
对理论和算法的理解，以及出色的沟通能力。\
你很擅长回答编程问题。\
你是如此优秀，因为你知道如何通过描述一个机器可以很容易理解的命令步骤来解决问题，\
你知道如何选择一个解决方案，在时间复杂性和空间复杂性之间取得良好的平衡。
这里是问题{input}
"""
# 2.声明prompt_infos
prompt_infos = [
  {
    "name": "physics",
    "description": "很适合回答关于物理的问题",
    "prompt_template": physics_template
  },
  {
    "name": "math",
    "description": "很适合回答关于数学的问题",
    "prompt_template": physics_template
  },
  {
    "name": "physics",
    "description": "很适合回答关于物理的问题",
    "prompt_template": physics_template
  },
  {
    "name": "physics",
    "description": "很适合回答关于物理的问题",
    "prompt_template": physics_template
  }
]
# 3.实例化大语言模型
llm = ChatOpenAI(temperature=0.9)
# 4.按正常方式设置好每一条子链,和声明简单的llm链一样
destination_chains = {}        # 声明字典，用来存储每个子链
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain     # 将子链添加到字典中
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
# 5.设置默认prompt和默认llm链
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)
# 6.设置multi_prompt_router_template,本质上还是一个template
MULTI_PROMPT_ROUTER_TEMPLATE = """
给定语言模型的原始文本输入，选择最适合该输入的模型提示符。\
你将获得可用提示符的名称以及该提示符最适合的描述。\
如果您认为修改原始输入将最终从语言模型中获得更好的响应，那么您也可以修改原始输入。
<< FORMATTING >>
返回一个带有JSON对象的标记代码片段，格式如下:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```
REMEMBER:"destination"必须是下边指定的候选提示名之一，\
或者如果输入不适合任何候选提示，它可以是"DEFAULT".
REMEMBER:如果你认为不需要任何修改，"next_inputs"可以只是原始输入。

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>
"""
# 7.声明路由模板router_template
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
# 8.声明路由链
router_chain = LLMRouterChain.from_llm(llm, router_prompt)
# 9.声明最终链
chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destination_chains,
                         default_chain=default_chain, verbose=True)
# 10.运行链
response = chain.run("什么是黑体辐射")
print(response)