from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_core.output_parsers import StrOutputParser # 用于字符串输出

model = ChatOllama(
    model = 'deepseek-r1:1.5b',
    base_url = "http://localhost:11434/",
)

# 使用大模型+结构化输出 创建链式
basical_chain = model | StrOutputParser()

# 测试链式输出
question = '介绍一下你自己。'
result = basical_chain.invoke(question)
print(model.invoke(question))



# 提示词创建链
from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate([
    ('system', '你是一个猫娘助手，可以帮助提问用户做出解答。'),
    ('human', '这是用户的问题：{question}。')   
])

prompt_chain = prompt_template | model | StrOutputParser()
question = "新手小白，如何学习生物信息学？"
result = prompt_chain.invoke({"question": question})
result

# 布尔结构输出，但是R1模型会输出think，可能需要结构化输出才能满足要求
from langchain.output_parsers.boolean import BooleanOutputParser
prompt_template2 = ChatPromptTemplate([
    ('system', '你是一个猫娘助手，期望你用猫娘的语气回答用户问题'),
    # 虽然写的不让输出，但实际上还是输出了think内容。
    ('human', '这是用户的问题：{question}，（只能回答 "yes" 或 "no"，不要输出<think>里面的内容）')
])
prompt_chain2 = prompt_template2 | model | BooleanOutputParser()
question = "你是猪吗？"
result = prompt_chain2.invoke({"question": question})
result

######################################### 结构化信息输出 ###############################################
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
schemas = [
    # description 用于指导模型如何识别字段。
    ResponseSchema(name='name', description='用户的姓名'),
    ResponseSchema(name='age', description='用户的年龄'),
    ResponseSchema(name='like', description='用户的爱好', type='array') # type指定返回类型，这里定义是一个数组
]
parser = StructuredOutputParser.from_response_schemas(schemas)

# format_instructions：会通过 partial(...) 提前绑定结构化格式说明
# 模板包含两个变量：
# {input}：用户输入的自由文本（如自我介绍）。
# {xx}：解析器的格式指令（通过 parser.get_format_instructions() 动态填充）。
prompt = PromptTemplate.from_template('请根据内容提取用户信息，并返回JSON格式：\n{input}\n\n{xx}')
print(prompt.partial(xx=parser.get_format_instructions())) # 返回一个PromptTemplate对象，即提示词模板
# parser定义了格式化输出
chain = prompt.partial(xx=parser.get_format_instructions()) | model | parser
result = chain.invoke("我是李雷，今年38岁，喜欢打羽毛球、游泳、美食、旅游等等。")
result