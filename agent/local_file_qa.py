#  基于VectorstoreIndexCreator创建的Q&A over Documents 
from langchain.chains import RetrievalQA        # 检索问答
from langchain.chat_models import ChatOpenAI    # 大语言模型
from langchain.document_loaders import CSVLoader  # csv加载器
from langchain.vectorstores import DocArrayInMemorySearch   # 向量存储
from IPython.display import display, Markdown      # 显示工具
from langchain.indexes import VectorstoreIndexCreator     # 向量存储索引创建器

# 1.讲csv文件放到文档加载器中
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file, encoding='utf-8')
# 2.创建向量索引对象
index = VectorstoreIndexCreator(
    vectorstore_cls = DocArrayInMemorySearch        # 可以选择其他向量存储器
    # embedding=embeddings,
).from_loaders([loader])
# 3.设置一个问题
query = """Please list all your shirts with sun protection \
in a table in markdown and summarize each one."""
# 4.讲问题放到向量索引中,得到基于文档的回答
response = index.query(query)
print(response)