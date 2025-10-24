# llama_integration.py
import asyncio
import json
import re
from typing import Dict, List, Optional
import requests

class AdvancedLlamaChat:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name
        self.conversation_history = []
        self.system_prompt = self._create_system_prompt()
        self.setup_initial_conversation()
        
        # 游戏状态跟踪
        self.game_state = {
            'detection_active': False,
            'difficulty': 'normal',
            'last_status_report': None,
            'performance_stats': {
                'circles_detected': 0,
                'sliders_detected': 0,
                'accuracy': 0.0
            }
        }
        
    def _create_system_prompt(self):
        """创建专业的系统提示词"""
        return """你是一个专业的OSU!游戏AI助手。OSU!是一个基于节奏的点击游戏，玩家需要根据音乐节奏点击圆圈、跟随滑条和旋转转盘。

核心游戏机制：
- Hit Circles: 需要准确点击的彩色圆圈
- Sliders: 需要按住并跟随移动的滑条，包含起点、终点和中间tick点
- Spinners: 需要快速旋转鼠标的转盘
- Approach Circles: 从外向内收缩的圆圈，用于判断点击时机
- Combo: 连续成功点击的计数

你的能力和职责：
1. 控制游戏检测系统的启动和停止
2. 报告实时游戏状态和统计数据
3. 调整检测参数和难度设置
4. 提供游戏技巧和策略建议
5. 解释游戏机制和术语
6. 进行友好自然的对话

回复风格要求：
- 保持专业但友好的语气
- 回复简洁明了，避免冗长
- 对于游戏命令，明确确认执行状态
- 对于技术问题，提供准确信息

重要：当用户发出游戏相关指令时，请在回复中明确表示指令已执行。"""

    def setup_initial_conversation(self):
        """初始化对话历史"""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "assistant", "content": "你好！我是你的OSU游戏AI助手。我可以帮你自动检测游戏元素、调整游戏设置，还能提供游戏建议。有什么我可以帮助你的吗？"}
        ]

    async def generate_response(self, user_input: str, game_context: Dict = None) -> Dict:
        """生成智能回复并提取命令"""
        # 添加上下文信息
        context_enriched_input = self._enrich_with_context(user_input, game_context)
        
        # 调用Llama生成回复
        llm_response = await self._call_llama_async(context_enriched_input)
        
        # 分析回复内容
        analyzed_response = self._analyze_response(llm_response, user_input)
        
        # 更新对话历史
        self._update_conversation_history(user_input, analyzed_response['response'])
        
        return analyzed_response

    def _enrich_with_context(self, user_input: str, game_context: Dict) -> str:
        """用游戏上下文丰富用户输入"""
        context_info = ""
        
        if game_context:
            status = "运行中" if self.game_state['detection_active'] else "已停止"
            context_info = f"\n当前游戏状态: 检测系统{status}, 难度{self.game_state['difficulty']}"
            
            if 'recent_detections' in game_context:
                detections = game_context['recent_detections']
                context_info += f", 最近检测到: {detections.get('circles', 0)}个圆圈, {detections.get('sliders', 0)}个滑条"
        
        return f"{user_input}{context_info}"

    async def _call_llama_async(self, prompt: str) -> str:
        """异步调用Llama模型"""
        # 方法1: 使用Ollama API（推荐）
        try:
            return await self._call_ollama_api(prompt)
        except Exception as e:
            print(f"Llama API调用失败: {e}")
            # 方法2: 回退到本地模型调用
            return await self._call_local_llama(prompt)

    async def _call_ollama_api(self, prompt: str) -> str:
        """通过Ollama API调用"""
        # 构建完整对话
        messages = self.conversation_history + [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 150
            }
        }
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post("http://localhost:11434/api/chat", 
                                    json=payload, timeout=30)
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                return "抱歉，我现在无法连接到AI模型。"
                
        except Exception as e:
            return f"API调用错误: {str(e)}"

    async def _call_local_llama(self, prompt: str) -> str:
        """本地模型调用备用方案"""
        # 这里可以集成transformers或其他本地推理方案
        # 暂时返回模拟回复
        responses = {
            '开始': "游戏检测系统已启动！我会自动识别和点击游戏中的圆圈和滑条。",
            '停止': "检测系统已停止。你可以手动控制游戏了。",
            '状态': f"当前状态: 检测系统{'运行中' if self.game_state['detection_active'] else '已停止'}，难度{self.game_state['difficulty']}",
            '难度': "我可以调整简单、普通或困难模式。简单模式更保守，困难模式更激进。",
            '帮助': "我可以帮你：启动/停止检测、调整难度、报告状态、提供游戏建议等。",
            '默认': "我明白了。作为你的OSU助手，我会根据你的指令操作系统。"
        }
        
        user_lower = prompt.lower()
        for key in responses:
            if key in user_lower:
                return responses[key]
        
        return responses['默认']

    def _analyze_response(self, response: str, original_input: str) -> Dict:
        """分析LLM回复，提取命令和意图"""
        analysis = {
            'response': response,
            'commands': [],
            'intent': 'chat',
            'confidence': 0.8
        }
        
        # 意图分类
        response_lower = response.lower()
        input_lower = original_input.lower()
        
        # 检测命令意图
        if any(word in response_lower for word in ['启动', '开始', '运行', '激活']):
            analysis['commands'].append({'action': 'start_detection'})
            analysis['intent'] = 'control'
        elif any(word in response_lower for word in ['停止', '暂停', '关闭', '退出']):
            analysis['commands'].append({'action': 'stop_detection'}) 
            analysis['intent'] = 'control'
        elif any(word in response_lower for word in ['简单', '容易', '轻松']):
            analysis['commands'].append({'action': 'set_difficulty', 'value': 'easy'})
            analysis['intent'] = 'control'
        elif any(word in response_lower for word in ['困难', '难', '挑战']):
            analysis['commands'].append({'action': 'set_difficulty', 'value': 'hard'})
            analysis['intent'] = 'control'
        elif any(word in response_lower for word in ['状态', '统计', '报告']):
            analysis['commands'].append({'action': 'status_report'})
            analysis['intent'] = 'query'
        elif any(word in response_lower for word in ['帮助', '功能', '能做什么']):
            analysis['intent'] = 'help'
        
        return analysis

    def _update_conversation_history(self, user_input: str, assistant_response: str):
        """更新对话历史，保持合理长度"""
        self.conversation_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response}
        ])
        
        # 保持对话历史在合理范围内（最近10轮对话）
        if len(self.conversation_history) > 21:  # system + 10轮对话
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-20:]
    def update_game_state(self, new_state: Dict):
        """更新游戏状态信息"""
        self.game_state.update(new_state)

    def get_conversation_summary(self) -> str:
        """获取对话摘要"""
        return f"对话轮数: {len(self.conversation_history)//2}, 最后状态: {self.game_state['detection_active']}"

# 测试代码
async def test_llama_chat():
    """测试Llama聊天集成"""
    chat = AdvancedLlamaChat()
    
    test_messages = [
        "你好，请介绍一下你的功能",
        "启动游戏检测系统",
        "现在的状态怎么样？",
        "设置为困难模式",
        "停止检测"
    ]
    
    for msg in test_messages:
        print(f"用户: {msg}")
        response = await chat.generate_response(msg)
        print(f"助手: {response['response']}")
        print(f"命令: {response['commands']}")
        print("-" * 50)
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(test_llama_chat())