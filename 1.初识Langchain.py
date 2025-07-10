import os

####################################### Langchain使用 ollama调用大模型方法 ########################################
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
client = ChatOllama(
    api_key = "ollama",
    base_url = "http://localhost:11434/",
    model = "deepseek-r1:14b",
    stream = True,
)

respose = client.invoke([
    SystemMessage(content="你是一个乐于助人的猫娘，根据用户问题，提供猫娘语气回答。"),
    HumanMessage(content="你好，请介绍一下你自己")
])
# 打印模型最终响应结果
for chunk in respose.content:
    print(chunk, end="", flush=True)


######################################## langchain格式 调用外部大模型 ###########################################
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv # 导入当前.env文件中的环境变量
from langchain.chat_models import init_chat_model # langchain自身初始化大模型函数
load_dotenv(override=True)
Deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
print(Deepseek_api_key) # 检测是否正确导入API-key
# 使用前提是必须之前下载了langchain对应模型的相关包，例如我下面使用deepseek，就需要下载langchain-deepseek。
# 这个包调用的是本机环境变量中的api-key，联网通过Langchain的框架连接到对应服务器厂商，然后调用大模型回复。
model = init_chat_model(model="deepseek-chat", model_provider="deepseek")
# 非流式输出
result = model.invoke("请说明一下Langchain-deepseek调用的模型是什么，我想知道和网站上的deepseek有什么区别吗")
result
print(result.content)
# 流式输出
result = model.stream("请介绍一下你自己")
# 逐块打印输出
for chunk in result:
    print(chunk.content, end="", flush=True)