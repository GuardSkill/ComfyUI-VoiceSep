import torch
import numpy as np
import soundfile as sf
import soxr
import tempfile
import os
import sys

# 确保当前目录在 Python 路径的最前面
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)
sys.path.insert(0, current_dir)
# 导入前再次确认路径
print(f"[ClearVoice Nodes] Current dir: {current_dir}")
print(f"[ClearVoice Nodes] sys.path[0]: {sys.path[0]}")

from clearvoice import ClearVoice


class ClearVoiceModelLoader:
    """加载 ClearVoice 模型的节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["MossFormer2_SS_16K"], {"default": "MossFormer2_SS_16K"}),
            }
        }
    
    RETURN_TYPES = ("CLEARVOICE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/clearvoice"
    
    def load_model(self, model_name):
        print(f"Initializing ClearVoice model: {model_name}...")
        
        # 保存当前工作目录
        original_cwd = os.getcwd()
        
        # 切换到 ClearVoice 所在的目录
        clearvoice_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(clearvoice_dir)
        
        try:
            cv_model = ClearVoice(
                task='speech_separation',
                model_names=[model_name]
            )
            print("Model initialized successfully")
            return (cv_model,)
        finally:
            # 恢复原来的工作目录
            os.chdir(original_cwd)


class ClearVoiceSpeechSeparation:
    """使用 ClearVoice 模型进行语音分离的节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CLEARVOICE_MODEL",),
                "audio": ("AUDIO",),
                "target_sample_rate": ([8000, 16000], {"default": 16000}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("separated_audio_1", "separated_audio_2")
    FUNCTION = "separate_audio"
    CATEGORY = "audio/clearvoice"
    
    def separate_audio(self, model, audio, target_sample_rate=16000):
        # 从 ComfyUI AUDIO 格式中提取数据
        # AUDIO 格式通常是: {"waveform": tensor, "sample_rate": int}
        waveform = audio["waveform"]  # shape: [batch, channels, samples]
        sample_rate = audio["sample_rate"]
        
        # 转换为 numpy 数组
        if isinstance(waveform, torch.Tensor):
            audio_np = waveform.squeeze(0).cpu().numpy()  # [channels, samples]
        else:
            audio_np = np.array(waveform).squeeze(0)
        
        # 如果是立体声，转换为单声道
        if len(audio_np.shape) > 1 and audio_np.shape[0] > 1:
            print("Converting stereo to mono...")
            audio_np = audio_np.mean(axis=0)  # 取平均值转为单声道
        elif len(audio_np.shape) > 1:
            audio_np = audio_np[0]  # 取第一个声道
        
        # 重采样到目标采样率
        if sample_rate != target_sample_rate:
            print(f"Resampling audio from {sample_rate} Hz to {target_sample_rate} Hz...")
            audio_np = soxr.resample(audio_np, sample_rate, target_sample_rate)
        
        # 创建临时文件保存音频
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_np, target_sample_rate)
        
        try:
            # 使用 ClearVoice 进行语音分离
            print("Separating audio...")
            output_wav_dict = model(input_path=temp_path, online_write=False)
            
            # 提取分离后的音频
            if isinstance(output_wav_dict, dict):
                key = next(iter(output_wav_dict))
                output_wav_list = output_wav_dict[key]
                output_wav_s1 = output_wav_list[0]
                output_wav_s2 = output_wav_list[1]
            else:
                output_wav_list = output_wav_dict
                output_wav_s1 = output_wav_list[0]
                output_wav_s2 = output_wav_list[1]
            
            # 转换为 ComfyUI AUDIO 格式
            # output_wav_s1 和 s2 的 shape 应该是 [1, samples]
            audio_1_tensor = torch.from_numpy(output_wav_s1).float().unsqueeze(0)  # [1, 1, samples]
            audio_2_tensor = torch.from_numpy(output_wav_s2).float().unsqueeze(0)  # [1, 1, samples]
            
            audio_output_1 = {
                "waveform": audio_1_tensor,
                "sample_rate": target_sample_rate
            }
            
            audio_output_2 = {
                "waveform": audio_2_tensor,
                "sample_rate": target_sample_rate
            }
            
            print("Audio separation completed successfully")
            return (audio_output_1, audio_output_2)
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "ClearVoiceModelLoader": ClearVoiceModelLoader,
    "ClearVoiceSpeechSeparation": ClearVoiceSpeechSeparation
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClearVoiceModelLoader": "Load ClearVoice Model",
    "ClearVoiceSpeechSeparation": "ClearVoice Speech Separation"
}