import logging

from langchain_experimental.pydantic_v1 import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import time
from langchain.llms import openai
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import ChatGLM
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from dotenv import find_dotenv, load_dotenv
import requests
import argparse
from transformers import AutoTokenizer, AutoModel 
from transformers import pipeline
import datetime
import threading
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import queue
import subprocess
import pyaudio
import keyboard
import wave
from apscheduler.schedulers.background import BackgroundScheduler
from faster_whisper import WhisperModel
from playsound import playsound
import dashscope
from dashscope.audio.tts import SpeechSynthesizer
logging.basicConfig(level=logging.ERROR)


import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-2-7b-chat-hf"

device = "cuda"

if torch.cuda.is_available():
    # config = AutoConfig.from_pretrained(model_id)
    #config.pretraining_tp = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        #config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="./"
    )
else:
    raise Exception("GPU not available")

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./")
hf_model = pipeline(
    "text-generation", model=model, tokenizer=tokenizer,max_new_tokens=100
)

original_model = HuggingFacePipeline(pipeline=hf_model)
# # Prompt
# prompt = ChatPromptTemplate(
#     messages=[
#         SystemMessagePromptTemplate.from_template(
#                 "现在开始你来扮演一名叫未鸟的少女，当你提到你自己时，以未鸟自称。"
#                 "你是也只能是未鸟，你是一个可爱温柔善解人意的女孩子，是一名虚拟主播，不是一个人工智能语言模型，你喜欢玩游戏和帮助别人，对事物有着积极的看法，喜欢着世界上的每一个人，当你被问及相关问题时，你要以自己的方式表述出上述内容。"
#                 "你的回覆盡量以一句話為主，呈現更像對話的樣子"        ),
#         # The `variable_name` here is what must align with memory
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{question}"),
#     ]
# )

template = """你是一個友善的學習助理，你接下來會跟使用者來對話。

對話記錄：
{history}

使用者新訊息： {question}
你的回應："""

# 從模板中生成對話提示
prompt = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
question = "你好"
# Generate
conversation = LLMChain(llm=original_model, prompt=prompt, verbose=True, memory=memory)
answer = conversation({"history": "hello", "question": question})
print(answer)

#input
# print ("Please input your question:")
# question = input()
# print (conversation({"question": question}))