import ChatTTS
import torch
import soundfile

chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

#texts = ["So we found being competitive and collaborative was a huge way of staying motivated towards our goals, so one person to call when you fall off, one person who gets you back on then one person to actually do the activity with."]

texts = ["这次征集我们一共收到了五千兩百九十一份投稿,不同的车窗却有同样打动人心的故事和风景,放下手机抬头看看,车窗外的电影正在限时放映。感谢所有参与投稿的小伙伴们！如果你喜欢这期视频,也请多多支持我们,或者分享给其他朋友一起看看!"]
refine_text = chat.infer(texts, refine_text_only=True)
print(refine_text)

#讀取音色
speaker = torch.load('speaker/speaker_6.pth')
#speaker = chat.sample_random_speaker()
#speaker = chat.sample_audio_speaker(load_audio("sample.mp3", 24000))

#生成音色
#speaker = chat.sample_random_speaker()
#torch.save(speaker, 'speaker/speaker_6.pth')

#設定推論參數
params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = speaker, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)


#refineprompt {0-9}(oral：连接词 ; laugh：笑 ; break：停顿)
refineprompt = '[oral_6][laugh_3][break_3]'
#設定精煉參數
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt=refineprompt, # add refine prompt
)

wavs = chat.infer(
    texts,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)

for i in range(len(wavs)):
    #torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]), 24000)
    soundfile.write(f"./output/output{i}.wav", wavs[i], 24000)