# transformer/download_data.py (重命名并修改此文件)
import os
import json

# 【关键】必须在导入 datasets 之前设置！
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("🌐 当前镜像端点:", os.environ['HF_ENDPOINT'])

try:
    from datasets import load_dataset
    print("📦 库加载成功...")
    
    # ✅ 修改这里：使用 opus100 的 zh-en 子集
    # 这个数据集已经是 Parquet 格式，支持新版加载器
    print("🚀 开始加载 Opus100 (en-zh) 数据集...")
    dataset = load_dataset("opus100", "en-zh")
    
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Opus100 的列名通常是 'translation'，结构和 IWSLT 一样
    splits = ['train', 'validation', 'test']
    
    # 注意：Opus100 的 test 集可能叫 'test' 或者包含在 train 里，具体看 HF 页面
    # 通常有 train, validation, test
    available_splits = [s for s in splits if s in dataset]
    if not available_splits:
        # 如果找不到 standard splits，尝试列出所有
        print(f"⚠️ 未找到标准分片，可用分片: {list(dataset.keys())}")
        available_splits = list(dataset.keys())[:3] # 取前三个

    for split in available_splits:
        file_path = os.path.join(data_dir, f"opus100_zh_en_{split}.jsonl")
        print(f"💾 正在保存 {split} 集到 {file_path} ...")
        
        count = 0
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in dataset[split]:
                # Opus100 的结构通常也是 {'translation': {'zh':..., 'en':...}}
                t = item['translation']
                line = json.dumps({"zh": t['zh'], "en": t['en']}, ensure_ascii=False)
                f.write(line + '\n')
                count += 1
        
        print(f"   ✅ 完成: {count} 条数据")
        
    print("\n🎉 成功！数据已保存到 ./data 目录。")
    print("💡 提示：请记得在后续的 data_loader.py 中将文件名改为 opus100_*.jsonl")

except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    import traceback
    traceback.print_exc()