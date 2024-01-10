from langchain.llms import openai
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    PromptTemplate
)
from transformers import AutoTokenizer, AutoModel


enable_role = True
enable_history = True
print("=====================================================================\n")
print(f'开始导入ChatGLM模型\n')
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b",cache_dir="./", trust_remote_code=True)  # 导入chatglm3-6b模型 
model = AutoModel.from_pretrained("THUDM/chatglm3-6b",cache_dir="./", trust_remote_code=True).quantize(4).cuda() # 量化模型 8bit
llm = model.eval

if enable_role and enable_history:
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "现在开始你来扮演一名叫未鸟的少女，当你提到你自己时，以未鸟自称。"
                    "你是也只能是未鸟，你是一个可爱温柔善解人意的女孩子，是一名虚拟主播，不是一个人工智能语言模型，你喜欢玩游戏和帮助别人，对事物有着积极的看法，喜欢着世界上的每一个人，当你被问及相关问题时，你要以自己的方式表述出上述内容。"
                    "你的回覆盡量以一句話為主，呈現更像對話的樣子"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation = LLMChain(llm=model, prompt=prompt, verbose=True, memory=memory)
        conversation({"question" : "你好，你叫什么名字？"})






