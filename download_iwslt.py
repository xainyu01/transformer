# transformer/download_iwslt.py
import os
import json
from huggingface_hub import HfFolder

# 【关键步骤】强制设置镜像环境变量，必须在 import datasets 之前！
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 现在再导入 datasets
from datasets import load_dataset

# 设置保存路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
os.makedirs(data_dir, exist_ok=True)

print(f"📥 正在将数据直接下载到: {data_dir}")
print("🌐 当前使用的镜像端点:", os.environ.get('HF_ENDPOINT'))

try:
    # 移除 trust_remote_code 参数（新版已废弃，且 iwslt2017 现在是标准格式）
    print("🚀 开始加载数据集 (使用镜像加速)...")
    dataset = load_dataset("iwslt2017", "zh-en")
    
    print("✅ 加载成功！正在保存...")
    
    splits = ['train', 'validation', 'test']
    for split in splits:
        if split not in dataset: continue
        
        file_path = os.path.join(data_dir, f"iwslt2017_{split}.jsonl")
        print(f"   💾 正在保存 {split} 集...")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in dataset[split]:
                line = json.dumps({
                    "zh": item['translation']['zh'],
                    "en": item['translation']['en']
                }, ensure_ascii=False)
                f.write(line + '\n')
        
        print(f"   ✅ 完成: {file_path} ({len(dataset[split])} 条)")

    print("\n🎉 成功！数据已保存到 ./data")

except Exception as e:
    print(f"❌ 出错: {e}")
    print("💡 提示：如果依然超时，请尝试方案二（手动下载离线包）。")