import ollama
import time
from datetime import datetime
import json
import numpy as np

class OllamaModelTester:
    def __init__(self, model_name='llama2-chinese'):
        self.model_name = model_name
        self.test_results = []
        self.test_prompts = [
            # 基础对话测试
            "你好，请做个自我介绍",
            "你的主要功能是什么？",
            
            # 情感理解测试
            "我今天感觉很难过",
            "我刚刚获得了一个重要奖项，很开心",
            "我最近压力很大，不知道该怎么办",
            
            # 知识测试
            "请解释下人工智能是什么",
            "请用简单的话解释下量子物理",
            
            # 创意测试
            "请写一首关于春天的短诗",
            "请讲一个简短的故事",
            
            # 复杂情境测试
            "我最近工作很忙，经常加班到很晚，感觉很疲惫，你有什么建议吗？",
            "我和朋友最近有些矛盾，应该如何处理？",
            
            # 逻辑推理测试
            "如果A比B大，B比C大，那么A和C的关系是什么？",
            "一个房间里有3只猫和2只狗，总共有多少条腿？"
        ]

    def test_basic_response(self):
        """测试基本响应功能"""
        print("\n=== 基本响应测试 ===")
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': '你好'}]
            )
            print("基本响应测试通过")
            print(f"响应内容: {response['message']['content']}")
            return True
        except Exception as e:
            print(f"基本响应测试失败: {str(e)}")
            return False

    def test_response_time(self, prompt):
        """测试响应时间"""
        start_time = time.time()
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            end_time = time.time()
            return end_time - start_time, response['message']['content']
        except Exception as e:
            print(f"响应时间测试失败: {str(e)}")
            return None, None

    def test_conversation_context(self):
        """测试对话上下文保持"""
        print("\n=== 上下文测试 ===")
        messages = [
            {'role': 'user', 'content': '我叫小明'},
            {'role': 'user', 'content': '你还记得我的名字吗？'}
        ]
        try:
            conversation = []
            for msg in messages:
                response = ollama.chat(
                    model=self.model_name,
                    messages=conversation + [msg]
                )
                conversation.append(msg)
                conversation.append({'role': 'assistant', 'content': response['message']['content']})
                print(f"用户: {msg['content']}")
                print(f"助手: {response['message']['content']}\n")
        except Exception as e:
            print(f"上下文测试失败: {str(e)}")

    def test_emotion_response(self):
        """测试情感响应"""
        print("\n=== 情感响应测试 ===")
        emotion_prompts = [
            "我今天很开心",
            "我感觉很沮丧",
            "我最近压力很大"
        ]
        for prompt in emotion_prompts:
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{
                        'role': 'system',
                        'content': '你是一个有同理心的AI助手，请对用户的情绪做出恰当的回应'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }]
                )
                print(f"情绪输入: {prompt}")
                print(f"AI回应: {response['message']['content']}\n")
            except Exception as e:
                print(f"情感响应测试失败: {str(e)}")

    def comprehensive_test(self):
        """综合测试"""
        print(f"\n开始测试模型: {self.model_name}")
        print("=" * 50)
        
        # 1. 基本功能测试
        if not self.test_basic_response():
            print("基本功能测试失败，停止后续测试")
            return
        
        # 2. 响应时间测试
        print("\n=== 响应时间测试 ===")
        response_times = []
        for prompt in self.test_prompts:
            response_time, response_content = self.test_response_time(prompt[:50] + "...")
            if response_time:
                response_times.append(response_time)
                self.test_results.append({
                    'prompt': prompt,
                    'response_time': response_time,
                    'response': response_content
                })
                print(f"提示词: {prompt[:50]}...")
                print(f"响应时间: {response_time:.2f}秒")
                print(f"回复: {response_content[:100]}...\n")
        
        if response_times:
            avg_time = np.mean(response_times)
            std_time = np.std(response_times)
            print(f"\n平均响应时间: {avg_time:.2f}秒")
            print(f"响应时间标准差: {std_time:.2f}秒")
        
        # 3. 上下文测试
        self.test_conversation_context()
        
        # 4. 情感响应测试
        self.test_emotion_response()
        
        # 保存测试结果
        self.save_test_results()

    def save_test_results(self):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'model_name': self.model_name,
            'test_time': timestamp,
            'results': self.test_results
        }
        
        filename = f"ollama_test_results_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n测试结果已保存到: {filename}")

def test_multiple_models():
    """测试多个模型"""
    models = [
        'llama2-chinese',
        'llama3-chinese',
        # 添加其他要测试的模型
    ]
    
    for model in models:
        tester = OllamaModelTester(model)
        print(f"\n开始测试模型: {model}")
        print("=" * 50)
        tester.comprehensive_test()

def main():
    """主函数"""
    try:
        # 测试单个模型
        tester = OllamaModelTester('llama3-chinese')
        tester.comprehensive_test()
        
        # 或测试多个模型
        # test_multiple_models()
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()