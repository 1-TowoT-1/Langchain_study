from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import asyncio

# 初始化大模型
model = ChatOllama(
    model = "glm4:9b",
    base_url = "http://localhost:11434/",
)

# 初始化输出解析器
parser = StrOutputParser()

# 初始化prompt提示词模板
prompt = ChatPromptTemplate([
    SystemMessage(content="你是一个智能助手，是生物信息学方面的专家，帮助用户解决问题。"),
    # MessagesPlaceholder 在对话链(conversation chain)中预留消息的位置。
    # 这个变量不是一个简单的字符串，可以是一组 ChatMessage 对象（例如 HumanMessage, AIMessage, SystemMessage 等）。
    MessagesPlaceholder(variable_name="chat_history")
])

# 创建链
basical_chain = prompt | model | parser

# 初始化对话历史
messages_list = []
print("* 输入 exit/quit 结束对话！")

async def chat_loop(messages_list):
    while True:
        user_query = input("👤 你：")
        if user_query.lower() in ['quit','exit']:
            break
        
        # 添加历史对话，添加的内容是HumanMessage对象
        messages_list.append(HumanMessage(content=user_query))
        
        # 调用模型流式生成响应
        ai_query_stream = basical_chain.astream({"chat_history": messages_list})  
        full_response = ""
        print("🤖 小智：", end="")
        async for chunk in ai_query_stream:
            # flush 是强制立即刷新输出缓冲区，确保内容实时显示在终端，而不是等待缓冲区满或换行时才输出。
            print(chunk, end="", flush=True)
            full_response += chunk
        print()

        # 添加AI回复到历史对话中
        messages_list.append(AIMessage(content=full_response))

        # 保留最后100条对话记录
        messages_list = messages_list[-100:]

# 运行异步主函数
asyncio.run(chat_loop(messages_list))