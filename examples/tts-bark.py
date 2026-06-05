from transformers import AutoProcessor, BarkModel
import os
os.environ["SUNO_ENABLE_MPS"] = "True"
os.environ["SUNO_OFFLOAD_CPU"] = "False"

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")


voice_preset = "v2/en_speaker_6"

inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)



audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

import scipy

sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)