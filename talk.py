# llama_chat.py
import subprocess
import json
import re

class LlamaChat:
    def __init__(self, model_path):
        self.model_path = model_path
        self.conversation_history = []
        
        # 系统提示词 - 定义AI的角色和能力
        self.system_prompt = """你是一个专业的OSU游戏AI助手。你能够：
1. 理解用户关于OSU游戏的指令和问题
2. 控制游戏检测系统的启动和停止
3. 报告游戏状态和统计数据
4. 提供游戏技巧和建议
5. 进行友好的日常对话

请保持回复简洁、有帮助，专注于游戏相关话题。"""
        
        self.setup_conversation()
    
    def setup_conversation(self):
        """初始化对话历史"""
        self.conversation_history = [            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "请介绍一下你的功能"},
            {"role": "assistant", "content": "我是你的OSU游戏AI助手！我可以帮你自动检测和点击游戏中的圆环，报告游戏状态，控制检测系统，还能聊天和提供游戏建议。有什么需要帮助的吗？"}
        ]
    
    def call_llama(self, prompt, max_tokens=150):
        """调用Llama模型生成回复"""
        try:
            # 添加用户新消息到对话历史
            self.conversation_history.append({"role": "user", "content": prompt})
            
            # 构建完整的对话上下文
            full_prompt = self._build_prompt()
            
            # 这里需要根据你的Llama部署方式调整
            # 以下是几种常见的调用方式：
            
            # 方式1: 使用ollama（如果使用ollama部署）
            result = self._call_ollama(full_prompt, max_tokens)
            
            # 方式2: 使用transformers库（如果直接加载模型）
            # result = self._call_transformers(full_prompt, max_tokens)
            
            # 方式3: 使用llama.cpp
            # result = self._call_llama_cpp(full_prompt, max_tokens)
            
            if result:
                self.conversation_history.append({"role": "assistant", "content": result})
                # 保持对话历史不会无限增长
                if len(self.conversation_history) > 10:
                    self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-8:]
            
            return result
            
        except Exception as e:
            print(f"Llama调用错误: {e}")
            return "抱歉，我现在遇到了一些技术问题。"
    
    def _build_prompt(self):
        """构建完整的提示词"""
        prompt = ""
        for msg in self.conversation_history:
            if msg["role"] == "system":
                prompt += f"系统: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"用户: {msg['content']}\n"
            else:
                prompt += f"助手: {msg['content']}\n"
        
        prompt += "助手: "
        return prompt
    
    def _call_ollama(self, prompt, max_tokens):
        """通过Ollama API调用（如果你使用Ollama）"""
        try:
            import requests
            payload = {
                "model": "llama3.1",  # 根据你的模型名称调整
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
            
            response = requests.post("http://localhost:11434/api/generate", 
                                   json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return "网络连接问题，请检查Ollama服务。"
                
        except ImportError:
            return "请安装requests库: pip install requests"
        except Exception as e:
            return f"Ollama调用失败: {e}"
    
    def _call_transformers(self, prompt, max_tokens):
        """直接使用transformers库调用"""
        # 这需要你已经有加载好的模型
        # 这里只是一个示例框架
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            # 假设模型已经加载到self.model和self.tokenizer
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取助手的回复部分
            return response.split("助手: ")[-1]
        except Exception as e:
            return f"模型调用错误: {e}"
    
    def get_game_command(self, response):
        """从LLM回复中提取游戏命令"""
        # 简单的关键词匹配提取命令
        response_lower = response.lower()
        
        commands = {}
        
        # 开始/停止检测
        if any(word in response_lower for word in ['开始', '启动', '打开检测']):
            commands['action'] = 'start_detection'
        elif any(word in response_lower for word in ['停止', '暂停', '关闭检测']):
            commands['action'] = 'stop_detection'
        
        # 状态报告请求
        if any(word in response_lower for word in ['状态', '报告', '统计']):
            commands['need_status'] = True
        
        # 难度设置
        if '简单' in response_lower:
            commands['difficulty'] = 'easy'
        elif '困难' in response_lower:
            commands['difficulty'] = 'hard'
        elif '普通' in response_lower:
            commands['difficulty'] = 'normal'
        
        return commands

# 使用示例
if __name__ == "__main__":
    # 初始化聊天助手
    chat = LlamaChat("llama3.1")  # 根据你的实际模型路径调整
    
    print("OSU AI聊天助手已启动！输入'退出'结束对话")
    
    while True:
        user_input = input("你: ")
        if user_input.lower() in ['退出', 'quit', 'exit']:
            break
        
        response = chat.call_llama(user_input)
        print(f"助手: {response}")
        
        # 提取命令
        commands = chat.get_game_command(response)
        if commands:
            print(f"提取到的命令: {commands}")

