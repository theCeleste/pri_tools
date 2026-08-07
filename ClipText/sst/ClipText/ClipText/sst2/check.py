import json

file_path = "/home/wrc/sstdp/CusText/CusText/sst2/embeddings/glove_840B-300d.txt"

print("--- 开始读取文件前几行 ---")
with open(file_path, 'r', encoding='utf-8') as f:
    # 1. 试读前500个字符，看看是不是 JSON 的开头
    sample = f.read(500)
    print("【文件前500个字符原始快照】:")
    print(sample)
    print("-" * 30)
    
    # 尝试判断它是否为 JSON
    try:
        json.loads(sample)
        print("💡 结论：这看起来是一个 JSON 格式的文件！")
    except json.JSONDecodeError:
        print("💡 结论：这不是一个完整的 JSON 格式（或者它只是普通的按行文本）。")

print("\n--- 尝试按行解析前 3 行 ---")
with open(file_path, 'r', encoding='utf-8') as f:
    for i in range(3):
        line = f.readline()
        if not line:
            print(f"第 {i+1} 行是空的。")
            continue
        
        parts = line.strip().split()
        print(f"【第 {i+1} 行】:")
        print(f"  - 这一行总共有多少个由空格隔开的元素: {len(parts)}")
        print(f"  - 开头第一个元素（通常是单词）: '{parts[0]}'")
        if len(parts) > 1:
            print(f"  - 后面紧跟的前3个切片（数字）: {parts[1:4]}...")