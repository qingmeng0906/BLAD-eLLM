import json
from tqdm import tqdm  
from openai import OpenAI
from transformers import RagTokenizer, RagRetriever
from datasets import Dataset
import torch
import time

class RAGEnhancedProcessor:
    def __init__(self, knowledge_base_path, openai_api_key="0", openai_base_url="http://0.0.0.0:8000/v1"):
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        
        # 初始化时间统计
        self.rag_init_time = 0
        self.rag_retrieval_times = []
        
        # 记录初始化开始时间
        init_start_time = time.time()
        
        # 加载知识库
        print("正在加载知识库...")
        kb_start = time.time()
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        kb_end = time.time()
        print(f"知识库加载耗时: {(kb_end - kb_start) * 1000:.1f}ms")
        
        # 初始化RAG组件
        print("正在加载RAG检索器...")
        rag_start = time.time()
        self.rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
        
        # 使用自定义知识库创建检索器
        self.rag_retriever = self.create_custom_retriever()
        rag_end = time.time()
        
        self.rag_init_time = (rag_end - rag_start) * 1000  # 转换为毫秒
        print(f"RAG检索器初始化耗时: {self.rag_init_time:.1f}ms")
        
        total_init_time = (time.time() - init_start_time) * 1000
        print(f"总初始化耗时: {total_init_time:.1f}ms")
        
    def load_knowledge_base(self, file_path):
        """加载知识库文件"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 确保返回的是列表格式
                if isinstance(data, dict):
                    # 如果是字典，转换为列表
                    return [{"title": key, "text": value} for key, value in data.items()]
                elif isinstance(data, list):
                    return data
                else:
                    # 如果是其他类型，转换为单项列表
                    return [{"title": "文档1", "text": str(data)}]
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                # 将文本按段落分割
                return [{"title": f"段落{i+1}", "text": line.strip()} for i, line in enumerate(lines) if line.strip()]
            else:
                raise ValueError("支持的文件格式：.json, .txt")
        except Exception as e:
            print(f"加载知识库失败：{str(e)}")
            return []
    
    def create_custom_retriever(self):
        """创建自定义检索器"""
        try:
            # 准备数据集格式
            if self.knowledge_base and isinstance(self.knowledge_base, list):
                # 确保数据格式正确
                formatted_data = []
                for item in self.knowledge_base:
                    if isinstance(item, dict):
                        title = item.get('title', '无标题')
                        text = item.get('text', item.get('content', ''))
                    else:
                        title = f"文档{len(formatted_data)+1}"
                        text = str(item)
                    
                    formatted_data.append({
                        "title": title,
                        "text": text
                    })
                
                # 创建数据集
                dataset = Dataset.from_list(formatted_data)
                
                # 这里需要构建索引，实际使用中可能需要更复杂的配置
                print(f"知识库包含 {len(formatted_data)} 个文档")
                return None  # 简化版本，使用文本匹配
            else:
                print("知识库为空或格式错误，使用默认检索器")
                return RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact", use_dummy_dataset=True)
                
        except Exception as e:
            print(f"创建检索器失败：{str(e)}")
            return RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact", use_dummy_dataset=True)
    
    def simple_text_retrieval(self, query, top_k=3):
        """简单的文本匹配检索（当自定义检索器创建失败时使用）"""
        if not self.knowledge_base or not isinstance(self.knowledge_base, list):
            return ""
        
        try:
            import re
            query_words = set(re.findall(r'\w+', query.lower()))
            
            scored_docs = []
            for doc in self.knowledge_base:
                # 修复：安全地处理不同数据类型
                if isinstance(doc, dict):
                    title = doc.get('title', '')
                    text = doc.get('text', doc.get('content', ''))
                elif isinstance(doc, str):
                    title = f"文档{len(scored_docs)+1}"
                    text = doc
                else:
                    title = f"文档{len(scored_docs)+1}"
                    text = str(doc)
                
                # 确保title和text都是字符串
                title = str(title) if title else ""
                text = str(text) if text else ""
                
                doc_text = f"{title} {text}".lower()
                doc_words = set(re.findall(r'\w+', doc_text))
                
                # 计算词汇重叠度
                overlap = len(query_words.intersection(doc_words))
                if overlap > 0:
                    # 统一格式化为字典
                    formatted_doc = {
                        'title': title,
                        'text': text
                    }
                    scored_docs.append((overlap, formatted_doc))
            
            # 按分数排序并返回top_k
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_docs = scored_docs[:top_k]
            
            context_parts = []
            for i, (score, doc) in enumerate(top_docs):
                title = doc.get('title', f'文档{i+1}')
                text = doc.get('text', '')
                context_parts.append(f"参考资料{i+1} - {title}:\n{text}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            print(f"文本检索异常：{str(e)}")
            # 添加调试信息
            print(f"知识库类型: {type(self.knowledge_base)}")
            if self.knowledge_base and hasattr(self.knowledge_base, '__len__'):
                print(f"知识库长度: {len(self.knowledge_base)}")
                if len(self.knowledge_base) > 0:
                    print(f"第一个元素类型: {type(self.knowledge_base[0])}")
                    print(f"第一个元素内容: {str(self.knowledge_base[0])[:100]}...")
            return ""
    
    def retrieve_context(self, query, top_k=3, max_context_length=1000):
        """检索相关上下文"""
        # 记录RAG检索开始时间
        retrieval_start = time.time()
        
        # 使用简单文本匹配检索
        context = self.simple_text_retrieval(query, top_k)
        
        # 控制上下文长度
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # 记录RAG检索结束时间
        retrieval_end = time.time()
        retrieval_time = (retrieval_end - retrieval_start) * 1000  # 转换为毫秒
        self.rag_retrieval_times.append(retrieval_time)
        
        return context, retrieval_time
    
    def create_enhanced_prompt(self, original_query, context, prompt_template=None):
        """创建增强提示词"""
        if prompt_template is None:
            if context:
                prompt_template = """请基于以下参考资料回答问题：

参考资料：
{context}

问题：{query}
"""
            else:
                return original_query
        
        return prompt_template.format(context=context, query=original_query)
    
    def process_single_case(self, case, retrieve_context=True, top_k=3, max_context_length=1000):
        """处理单个案例"""
        original_input = case['input']
        context = ""
        rag_time = 0
        
        if retrieve_context:
            context, rag_time = self.retrieve_context(original_input, top_k, max_context_length)
        
        enhanced_prompt = self.create_enhanced_prompt(original_input, context)
        
        messages = [{"role": "user", "content": enhanced_prompt}]
        
        # 记录API调用时间
        api_start = time.time()
        try:
            response = self.client.chat.completions.create(
                model="test",
                messages=messages
            )
            api_end = time.time()
            api_time = (api_end - api_start) * 1000  # 转换为毫秒
            
            result = {
                "predict": response.choices[0].message.content.strip(),
                "enhanced_prompt": enhanced_prompt,
                "retrieved_context": context,
                "context_length": len(context),
                "has_context": bool(context),
                "rag_retrieval_time_ms": rag_time,
                "api_call_time_ms": api_time,
                "total_time_ms": rag_time + api_time
            }
            
        except Exception as e:
            api_end = time.time()
            api_time = (api_end - api_start) * 1000  # 转换为毫秒
            print(f"\n处理异常：{str(e)}")
            result = {
                "predict": "error",
                "enhanced_prompt": enhanced_prompt,
                "retrieved_context": context,
                "context_length": len(context),
                "has_context": bool(context),
                "rag_retrieval_time_ms": rag_time,
                "api_call_time_ms": api_time,
                "total_time_ms": rag_time + api_time,
                "error": str(e)
            }
        
        return result
    
    def get_rag_statistics(self):
        """获取RAG时间统计信息（毫秒）"""
        if not self.rag_retrieval_times:
            return {
                "total_retrievals": 0,
                "total_retrieval_time_ms": 0,
                "avg_retrieval_time_ms": 0,
                "min_retrieval_time_ms": 0,
                "max_retrieval_time_ms": 0,
                "init_time_ms": self.rag_init_time
            }
        
        return {
            "total_retrievals": len(self.rag_retrieval_times),
            "total_retrieval_time_ms": sum(self.rag_retrieval_times),
            "avg_retrieval_time_ms": sum(self.rag_retrieval_times) / len(self.rag_retrieval_times),
            "min_retrieval_time_ms": min(self.rag_retrieval_times),
            "max_retrieval_time_ms": max(self.rag_retrieval_times),
            "init_time_ms": self.rag_init_time
        }
    
    def debug_knowledge_base(self):
        """调试知识库格式"""
        print(f"\n🔍 知识库调试信息:")
        print(f"  知识库类型: {type(self.knowledge_base)}")
        
        if self.knowledge_base is None:
            print("  知识库为空(None)")
            return
            
        if hasattr(self.knowledge_base, '__len__'):
            print(f"  知识库长度: {len(self.knowledge_base)}")
        else:
            print("  知识库没有长度属性")
            return
        
        if isinstance(self.knowledge_base, list) and self.knowledge_base:
            # 显示前3个元素
            for i in range(min(3, len(self.knowledge_base))):
                item = self.knowledge_base[i]
                print(f"  项目 {i+1} 类型: {type(item)}")
                print(f"  项目 {i+1} 内容: {str(item)[:100]}...")
                if isinstance(item, dict):
                    print(f"  项目 {i+1} 键: {list(item.keys())}")
        elif isinstance(self.knowledge_base, dict):
            print("  知识库是字典格式，键:")
            keys = list(self.knowledge_base.keys())
            for i, key in enumerate(keys[:3]):
                print(f"    键 {i+1}: {key}")
        else:
            print(f"  知识库是 {type(self.knowledge_base)} 类型")

# 使用示例 - 在这里指定你的知识库文件路径
KNOWLEDGE_BASE_PATH = "/home/lab1015/programmes/sjc/LLM/RAG2.json"  # 修改为你的知识库路径

# 记录整体开始时间
total_start_time = time.time()

processor = RAGEnhancedProcessor(KNOWLEDGE_BASE_PATH)

# 添加调试信息
processor.debug_knowledge_base()

with open('/home/lab1015/programmes/sjc/LLM/ragqwen.json', 'r+', encoding='utf-8') as f:
    test_data = json.load(f)
    
    print(f"\n开始处理 {len(test_data)} 个案例...")
    processing_start_time = time.time()
    
    for index, case in enumerate(tqdm(test_data, desc='处理进度', unit='条')):
        result = processor.process_single_case(
            case, 
            retrieve_context=True,
            top_k=3,
            max_context_length=800
        )
        
        test_data[index].update(result)

    processing_end_time = time.time()
    
    f.seek(0)
    json.dump(test_data, f, indent=2, ensure_ascii=False)
    f.truncate()

total_end_time = time.time()

# 输出详细的时间统计
print("\n" + "="*50)
print("处理完成，结果已保存")
print("="*50)

# RAG统计信息
rag_stats = processor.get_rag_statistics()
print(f"\n📊 RAG模块时间统计:")
print(f"  初始化时间: {rag_stats['init_time_ms']:.1f}ms")
print(f"  检索次数: {rag_stats['total_retrievals']}")
print(f"  总检索时间: {rag_stats['total_retrieval_time_ms']:.1f}ms")
print(f"  平均检索时间: {rag_stats['avg_retrieval_time_ms']:.1f}ms")
print(f"  最快检索时间: {rag_stats['min_retrieval_time_ms']:.1f}ms")
print(f"  最慢检索时间: {rag_stats['max_retrieval_time_ms']:.1f}ms")

# 整体统计
total_processing_time = (processing_end_time - processing_start_time) * 1000  # 转换为毫秒
total_program_time = (total_end_time - total_start_time) * 1000  # 转换为毫秒

print(f"\n⏱️  整体时间统计:")
print(f"  数据处理时间: {total_processing_time:.1f}ms")
print(f"  程序总运行时间: {total_program_time:.1f}ms")
print(f"  平均每个案例处理时间: {total_processing_time/len(test_data):.1f}ms")

# 计算API调用时间统计
api_times = [case.get('api_call_time_ms', 0) for case in test_data if 'api_call_time_ms' in case]
if api_times:
    print(f"\n🌐 API调用时间统计:")
    print(f"  总API调用时间: {sum(api_times):.1f}ms")
    print(f"  平均API调用时间: {sum(api_times)/len(api_times):.1f}ms")
    print(f"  最快API调用: {min(api_times):.1f}ms")
    print(f"  最慢API调用: {max(api_times):.1f}ms")

# 计算成功率
successful_cases = sum(1 for case in test_data if case.get('predict', '') != 'error')
success_rate = successful_cases / len(test_data) * 100
cases_with_context = sum(1 for case in test_data if case.get('has_context', False))
context_rate = cases_with_context / len(test_data) * 100

print(f"\n📈 处理结果统计:")
print(f"  总处理案例: {len(test_data)}")
print(f"  成功处理案例: {successful_cases}")
print(f"  成功率: {success_rate:.1f}%")
print(f"  包含检索上下文的案例: {cases_with_context}")
print(f"  上下文覆盖率: {context_rate:.1f}%")

# RAG与API时间对比
if api_times and rag_stats['total_retrievals'] > 0:
    print(f"\n🔍 RAG vs API时间对比:")
    print(f"  RAG检索平均时间: {rag_stats['avg_retrieval_time_ms']:.1f}ms")
    print(f"  API调用平均时间: {sum(api_times)/len(api_times):.1f}ms")
    ratio = (sum(api_times)/len(api_times)) / rag_stats['avg_retrieval_time_ms']
    print(f"  API/RAG时间比例: {ratio:.1f}x")

print("\n" + "="*50)