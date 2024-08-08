# 导入两个用于解析结果的包Response和StructureOutputParser
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# 修改为你自己配置的OPENAI_API_KEY
api_key = "evllm"

# 修改为你启动api-for-open-llm项目所在的服务地址和端口
api_url = "http://localhost:8000/v1"

# modal= "baichuan2-13b-chat"
modal= "chatglm"


# 这里我们想得到一个字典格式的回复
# {"name": "zhangsan", "age": "15", "id": "123456"}
# 1. 设置这三个字段的结构信息
name_schema = ResponseSchema(name="name", description="string类型的,用户的姓名")
age_schema = ResponseSchema(name="age", description="int类型的,用户的年龄")
id_schema = ResponseSchema(name="id", description="string类型的,用户的id")
# 2.将这三个字段统一放入响应结构中
response_schemas = [name_schema,
                   age_schema,
                   id_schema]
# 3. 将响应结构导入到解析器中，得到输出解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# 4. 从输出解析器output_parser中得到  传给LLM的指令格式
format_instructions = output_parser.get_format_instructions()
# 5. 设置模板，模板中最后要加入指令格式format_instructions
template = """下边的text是一个学生的基本情况,对其提取以下信息:\
name:这个学生的姓名是什么?
age:这个学生的年龄是多少?
id:这个学生的学号是多少?
text:{text}
{format_instructions}"""
# 一段人类世界的普通文字
text = """这是ai大学的一个学生,他来自北京,\
现在他刚刚成年了,今天他准备去买一个蛋糕,\
学校门口的保安让他报一下他的学号,他就说他的学号是194050225,\
到了蛋糕店,老板需要把他的名字写在蛋糕上,因此他就告诉老板,他的名字叫李逍遥.\
最后他成功的买上了蛋糕,回学校了."""
# 6. 加载模板到prompt中
prompt = ChatPromptTemplate.from_template(template=template)
# 7. 给提示中的变量赋值，得到最终要传给LLM的消息
messages = prompt.format_messages(text=text, format_instructions=format_instructions)
print("[message]:", messages)
# 8.实例化大语言模型
# llm = ChatOpenAI(temperature=0.9)
llm = ChatOpenAI(model_name=modal,openai_api_key=api_key,openai_api_base=api_url)
# 9.将消息输入给大语言模型
response = llm(messages)        # 注意此时得到的结果仍然是str类型的
# 10.使用解析器来解析这个结果，将其转变为字典类型
output_dict = output_parser.parse(response.content)
# 输出字典形式的结果
print(type(output_dict), output_dict)