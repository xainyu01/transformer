# transformer/test_import.py
import sys
import os

# 打印当前工作目录
print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

# 尝试直接导入
try:
    from model.multi_head_attention import MultiHeadAttention
    print("SUCCESS: Imported MultiHeadAttention from model.multi_head_attention")
except Exception as e:
    print("ERROR: Failed to import MultiHeadAttention")
    print("Exception:", str(e))
    print("File exists:", os.path.exists("model/multi_head_attention.py"))
    print("File content (first 100 chars):", open("model/multi_head_attention.py", "r").read(100))