import requests
import time
import json

class OllamaTest:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.test_prompt = "你好，请用简短的话回答我"
        
    def check_service(self):
        """检查Ollama服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                print("✓ Ollama服务连接正常")
                return True
            else:
                print(f"✗ Ollama服务响应异常，状态码: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("✗ 无法连接到Ollama服务，请检查服务是否启动")
            return False
        except Exception as e:
            print(f"✗ 检查服务时出错: {str(e)}")
            return False

    def list_models(self):
        """列出所有可用模型"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                print("\n可用的模型:")
                for model in models.get("models", []):
                    print(f"- {model['name']}")
                return models.get("models", [])
            return []
        except Exception as e:
            print(f"获取模型列表时出错: {str(e)}")
            return []

    def test_chat(self, model_name):
        """测试与指定模型的对话"""
        print(f"\n测试与模型 {model_name} 的对话...")
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": self.test_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                elapsed_time = time.time() - start_time
                result = response.json()
                print(f"✓ 对话测试成功 (用时: {elapsed_time:.2f}秒)")
                print(f"提问: {self.test_prompt}")
                print(f"回答: {result['response']}")
                return True
            else:
                print(f"✗ 对话测试失败，状态码: {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            print("✗ 对话请求超时")
            return False
        except Exception as e:
            print(f"✗ 对话测试出错: {str(e)}")
            return False

    def run_comprehensive_test(self):
        """运行完整的测试流程"""
        print("开始Ollama功能测试...\n")
        
        # 测试1: 检查服务
        print("测试1: 检查Ollama服务")
        if not self.check_service():
            print("\n测试终止: Ollama服务未就绪")
            return False
            
        # 测试2: 获取模型列表
        print("\n测试2: 获取可用模型")
        models = self.list_models()
        if not models:
            print("\n测试终止: 未找到可用模型")
            return False
            
        # 测试3: 与每个模型进行对话测试
        print("\n测试3: 模型对话测试")
        test_results = {}
        for model in models:
            model_name = model['name']
            test_results[model_name] = self.test_chat(model_name)
            time.sleep(1)  # 避免请求太频繁
            
        # 输出测试总结
        print("\n测试总结:")
        print(f"- 服务状态: 正常")
        print(f"- 可用模型数量: {len(models)}")
        print("- 对话测试结果:")
        for model_name, success in test_results.items():
            status = "✓ 成功" if success else "✗ 失败"
            print(f"  • {model_name}: {status}")
            
        return all(test_results.values())

    def test_model_performance(self, model_name, test_cases=None):
        """测试模型的性能和响应质量"""
        if test_cases is None:
            test_cases = [
                "你好，请做个自我介绍",
                "1+1等于多少？请只回答数字",
                "请用一句话描述春天",
                "你能理解中文吗？请回答'能'或'不能'"
            ]

        print(f"\n对模型 {model_name} 进行性能测试...")
        results = []
        
        for case in test_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": case,
                        "stream": False
                    }
                )
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "prompt": case,
                        "response": result["response"],
                        "time": elapsed_time
                    })
                    print(f"\n问题: {case}")
                    print(f"回答: {result['response']}")
                    print(f"用时: {elapsed_time:.2f}秒")
                else:
                    print(f"\n问题: {case}")
                    print(f"测试失败: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"\n问题: {case}")
                print(f"测试出错: {str(e)}")
                
            time.sleep(1)  # 避免请求太频繁
            
        # 计算性能指标
        if results:
            avg_time = sum(r["time"] for r in results) / len(results)
            print(f"\n性能总结:")
            print(f"- 平均响应时间: {avg_time:.2f}秒")
            print(f"- 成功率: {len(results)}/{len(test_cases)}")
        
        return results

def main():
    tester = OllamaTest()
    
    # 运行基础测试
    if tester.run_comprehensive_test():
        print("\n基础测试完成，是否要进行详细的性能测试？(y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            # 获取可用模型
            models = tester.list_models()
            if models:
                print("\n选择要测试的模型（输入序号）：")
                for i, model in enumerate(models, 1):
                    print(f"{i}. {model['name']}")
                    
                try:
                    model_idx = int(input()) - 1
                    if 0 <= model_idx < len(models):
                        model_name = models[model_idx]['name']
                        tester.test_model_performance(model_name)
                    else:
                        print("无效的选择")
                except ValueError:
                    print("无效的输入")
    
    print("\n测试完成")

if __name__ == "__main__":
    main()