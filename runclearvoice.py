
from clearvoice import ClearVoice
import os
import time
import soundfile as sf
import soxr
# 初始化语音分离模型

T1=time.time()
cv_ss = ClearVoice(
    task='speech_separation',
    model_names=['MossFormer2_SS_16K']
)
print("Model init Time used", time.time() - T1)



input_path = '/root/lisiyuan/Pyprojects/speech_separation/ComfyUI_temp_axihh_00005_.flac'
output_dir = 'output_separated'
output_wav_dict = cv_ss(input_path=input_path, online_write=False)
if isinstance(output_wav_dict, dict):
    key = next(iter(output_wav_dict))
    output_wav_list = output_wav_dict[key]
    output_wav_s1 = output_wav_list[0]
    output_wav_s2 = output_wav_list[1]
else:
    output_wav_list = output_wav_dict
    output_wav_s1 = output_wav_list[0]
    output_wav_s2 = output_wav_list[1]
sf.write('separated_s1.wav', output_wav_s1[0,:], 16000)
sf.write('separated_s2.wav', output_wav_s2[0,:], 16000)

# os.makedirs(output_dir, exist_ok=True)
# # 分离语音并自动保存
# cv_ss(
#     input_path=input_path,
#     online_write=True,
#     output_path=output_dir
# )
# print("infer Time used", time.time() - T1)


# T1=time.time()
# # 处理混合语音文件
# input_path = '《沈园外-阿YueYue 戾格 小田音乐社》.mp3'
# output_dir = 'output_separated'
# os.makedirs(output_dir, exist_ok=True)
# # 分离语音并自动保存
# cv_ss(
#     input_path=input_path,
#     online_write=True,
#     output_path=output_dir
# )
# print("infer Time used", time.time() - T1)

# T1=time.time()

input_path = '/root/lisiyuan/Pyprojects/speech_separation/《沈园外-阿YueYue 戾格 小田音乐社》.mp3'
T1=time.time()
# 处理混合语音文件
resolution=16000      #resolution=8000
audio, sr = sf.read(input_path)
T1=time.time()
if sr != resolution:
    print(f"Resampling audio from {sr} Hz to {resolution}Hz...")
    # soxr 是最快的重采样库
    # 如果是立体声，转换为单声道
    if len(audio.shape) > 1:
        print(f"Converting stereo to mono...")
        audio = audio.mean(axis=1)  # 取平均值转为单声道
    audio = soxr.resample(audio, sr, resolution)
    temp_file = 'temp_resampled_processed.wav'
    sf.write(temp_file, audio, resolution)
    input_path = temp_file
print(f"Resampling Done")
print("ResamplingTime used",time.time()-T1)

output_dir = 'output_separated'
os.makedirs(output_dir, exist_ok=True)
# 分离语音并自动保存
cv_ss(
    input_path=input_path,
    online_write=True,
    output_path=output_dir
)
print("infer Time used", time.time() - T1)

