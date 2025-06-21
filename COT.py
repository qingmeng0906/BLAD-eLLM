import json
from tqdm import tqdm  
from openai import OpenAI

# 定义需要追加的指令文本
INSTRUCTION = """
请根据以上网络流量的特征数据判断是否异常。正常输出:0 ; 异常输出:1。请按照以下步骤思考并综合评估：
1.理解各个字段的含义
2.关联特征分析，识别可能的异常迹象
3.与正常流量模型对比，考虑是否可能误报
4.综合所有分析结果，给出最终判断
"""

client = OpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")

with open('', 'r+', encoding='utf-8') as f:
    test_data = json.load(f)
    
    for index, case in enumerate(tqdm(test_data, desc='处理进度', unit='条')):
        # 在原始输入后添加指令
        augmented_input = f"{case['input']}\n\n{INSTRUCTION}"
        
        messages = [{
            "role": "user", 
            "content": augmented_input
        }]
        
        try:
            response = client.chat.completions.create(
                model="test",
                messages=messages
            )
            test_data[index]["predict"] = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n处理异常：{str(e)}")  
            test_data[index]["predict"] = "error"

    f.seek(0)
    json.dump(test_data, f, indent=2, ensure_ascii=False)
    f.truncate()

print("\n处理完成，结果已保存")