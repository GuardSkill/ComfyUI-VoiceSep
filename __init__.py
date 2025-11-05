"""
ClearVoice Speech Separation Nodes for ComfyUI
"""

import sys
import os

# 将当前文件夹添加到 Python 路径的最前面，确保优先导入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))

# 移除可能存在的旧路径，避免重复
if current_dir in sys.path:
    sys.path.remove(current_dir)
sys.path.insert(0, current_dir)

# 打印调试信息
print(f"[ClearVoice] Adding to sys.path: {current_dir}")
print(f"[ClearVoice] Checking utils dir: {os.path.join(current_dir, 'utils')}")
print(f"[ClearVoice] Utils exists: {os.path.exists(os.path.join(current_dir, 'utils'))}")

# 验证关键文件是否存在
utils_decode = os.path.join(current_dir, 'utils', 'decode.py')
if os.path.exists(utils_decode):
    print(f"[ClearVoice] ✓ Found utils/decode.py")
else:
    print(f"[ClearVoice] ✗ Missing utils/decode.py at {utils_decode}")

from .clearvoice_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# 可选：添加版本信息和描述
WEB_DIRECTORY = "./web"
__version__ = "1.0.0"