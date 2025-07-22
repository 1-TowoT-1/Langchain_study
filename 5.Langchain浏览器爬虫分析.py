from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor,create_tool_calling_agent
from langchain import hub

# 启动浏览器
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_core.output_parsers import StrOutputParser

sync_browser = create_sync_playwright_browser()
browser_tool = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = browser_tool.get_tools()
tools

prompt = hub.pull("hwchase17/openai-tools-agent")
print(prompt)

model = ChatOllama(
    model = "llama3.1:8b",
    base_url = "http://localhost:11434/",
)

parser = StrOutputParser()

# 创建agent
agent = create_tool_calling_agent(model,tools,prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
command = {'input': '访问https://github.com/fufankeji/MateGen/blob/main/README_zh.md 并帮我总结一下这个网站的内容'}
# 访问网站调用的外部工具PlayWrightBrowserToolkit。外部工具获取的信息再返回给大模型。然后大模型再输出给用户。
response = agent_executor.invoke(command)
print(response)


# 由于电脑太次了，直接调用api对应的deepseek全量模型分析
from dotenv import load_dotenv 
from langchain.chat_models import init_chat_model
load_dotenv(override=True)

deepseek_model = init_chat_model("deepseek-chat", model_provider="deepseek")
deepseek_agent = create_tool_calling_agent(deepseek_model,tools,prompt)
deepseek_agent_executor = AgentExecutor(agent=deepseek_agent, tools=tools, verbose=True)
command = {'input': '访问https://github.com/fufankeji/MateGen/blob/main/README_zh.md 并帮我总结一下这个网站的内容'}
deepseek_response = deepseek_agent_executor.invoke(command)
print(deepseek_response)


command = {'input': '访问https://github.com/fufankeji/MateGen/blob/main/README_zh.md 并帮我总结一下这个网站的内容'}
deepseek_response = deepseek_agent_executor.invoke(command)
print(deepseek_response)

# 下面的结果只是简单的介绍了一下网址对应的页面上的介绍，如果需要访问文件，读取，需要进一步的增加tool功能，使大模型可以自己判断调用。
command = {'input': '访问https://github.com/1-TowoT-1/MedicalData_VisualizationPlatform 并帮助我总结这个项目的结构，分析下组成何总结下这个项目的内容。'}
deepseek_response = deepseek_agent_executor.invoke(command)
print(deepseek_response)
print(deepseek_response['output'])

# 想法是直接让大模型总结bismark软件的内部运行步骤，由于bismark源码非常大，直接卡在了分析界面，在设置agent的过程中没有设置max_tokens，遇到超大文本分析，比较好的方法还是切片调用，逐段分析。
command = {'input': '访问https://raw.githubusercontent.com/FelixKrueger/Bismark/refs/heads/master/bismark 总结下bismark这个软件的内部运行步骤是怎样的。'}
deepseek_response = deepseek_agent_executor.invoke(command)
print(deepseek_response)
print(deepseek_response['output'])