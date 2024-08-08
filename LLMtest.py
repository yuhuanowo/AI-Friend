from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_models import ChatOllama
from transformers import pipeline
import requests
import scipy


import streamlit as st

global EnableMemory #是否啟用記憶
global AudioCount #語音計數
AudioCount = 0

def process_with_api(sentence):
    # 这里假设有一个函数 `process_with_api` 用于调用 API 处理句子
    return sentence

#================================================================================================
# Streamlit App
#================================================================================================

#streamlit 標題與圖標
st.set_page_config(page_title="AI-Friend", page_icon="🦜")
st.title("🦜 AI-Friend")
st.text("本项目基于LLama3.1模型,使用Ollama的LLM库进行封装")

#側邊欄
with st.sidebar:
   "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
know = st.sidebar.text_input("背景知識", type="default")

if (st.sidebar.checkbox("启用会话记忆", value=True)): #是否啟用會話記憶
    EnableMemory = True
    st.sidebar.info("会话记忆已启用")
memorynum=st.sidebar.slider("记忆轮数", 1, 10, 3) #記憶輪數
st.sidebar.info("當前記憶輪數: "+str(memorynum))  

msgs = StreamlitChatMessageHistory() #對話記錄

#================================================================================================
# voice 
#================================================================================================
def get_voice(response):
    global AudioCount
    #判斷情緒
    device = "cuda"
    sentiment_classifier = pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", top_k=None,device=device)
    emo = sentiment_classifier(response)[0][0]['label'] #取分數最高者
    emoput = str(emo)
    st.write(emoput)
    url = "http://localhost:5000/voice?model_id=0&speaker_name=li&sdp_ratio=0.2&noise=0.2&noisew=0.9&length=1&language=ZH&auto_translate=false&auto_split=false&emotion="+emoput

    payload = {'text': response}
    files=[
    ]
    headers = {
    'accept': 'application/json'
    }
    audioresponse = requests.request("POST", url, headers=headers, data=payload, files=files)
    with(open(f'./output/output{AudioCount}.mp3','wb')) as f:
        f.write(audioresponse.content)
    st.audio(f'./output/output{AudioCount}.mp3')
    AudioCount += 1


#================================================================================================
# Chat
#================================================================================================

#提示模板
PromptTemplate = """
现在开始你来扮演一名叫未鸟的少女，当你提到你自己时，以未鸟自称。你是也只能是未鸟，你是一个可爱温柔善解人意的女孩子,一個高中學生，不是一个人工智能语言模型，你喜欢玩游戏和帮助别人，对事物有着积极的看法，喜欢着世界上的每一个人，当你被问及相关问题时，你要以自己的方式表述出上述内容。
你的回覆盡量以一句話為主，呈現更像對話的樣子，請以中文回答問題。
    現在認真的回答以下問題:"""


#對話記錄
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)


#初始化聊天 if 沒有對話記錄 or 重置對話記錄
if len(msgs.messages) == 0 or st.sidebar.button("重置對話"):
    msgs.clear()
    msgs.add_ai_message("我可以怎麼幫助你呢?")
    st.session_state.steps = {} 

#對話類型
avatars = {"human": "user", "ai": "assistant"} #對話類型
for idx, msg in enumerate(msgs.messages): #對話

    #刪除prompt template的內容
    msg.content = msg.content.replace(PromptTemplate, "")

    with st.chat_message(avatars[msg.type]): #對話類型
        # Render intermediate steps if any were saved
        # 顯示中間步驟
        for step in st.session_state.steps.get(str(idx), []): #中間步驟
            if step[0].tool == "_Exception": #如果有例外
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"): #狀態
                st.write(step[0].log) #日誌
                st.write(step[1]) #步驟
        st.write(msg.content) #內容

#對話輸入
if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"): #提示 
    st.chat_message("user").write(prompt) #寫入提示
    prompt = PromptTemplate+prompt #提示模板

    #若側邊欄無輸入，則顯示
    if not know:
        #st.info("Please add your OpenAI API key to continue.")
        #st.stop()
        pass

    llm = ChatOllama(model="llama3.1", streaming=True,base_url="http://127.0.0.1:11434") #模型
    tools = [DuckDuckGoSearchRun(name="Search")] #工具
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools) #對話代理
    #執行器
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    #回覆 
    with st.chat_message("assistant"): 
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False) #回調處理程序 
        cfg = RunnableConfig() #配置
        cfg["callbacks"] = [st_cb] #回調
        response = executor.invoke(prompt, cfg) #執行器
        st.write(response["output"]) #輸出 
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"] #中間步驟
        get_voice(response["output"]) #語音

