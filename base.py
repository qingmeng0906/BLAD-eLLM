import json
import time
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")

with open('', 'r+', encoding='utf-8') as f:
    test_data = json.load(f)
    
    # 记录处理开始时间
    start_time = time.time()
    
    for index, case in enumerate(tqdm(test_data, desc='处理进度', unit='条')):
        messages = [{
            "role": "user", 
            "content": f"{case['input']}"
        }]
        
        try:
            response = client.chat.completions.create(
                model="qwen",
                messages=messages,
                temperature=0,
                max_tokens=1,
                stop=["0", "1"]
            )
            test_data[index]["predict"] = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n处理异常：{str(e)}")
            test_data[index]["predict"] = "error"
    
    # 计算纯处理时间（毫秒）
    total_ms = (time.time() - start_time) * 1000  # 秒转毫秒

    # 回写文件
    f.seek(0)
    json.dump(test_data, f, indent=2, ensure_ascii=False)
    f.truncate()

print(f"\n处理完成，结果已保存。模型处理耗时：{total_ms:.0f}毫秒")  # 保留整数毫秒