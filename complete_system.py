# complete_system.py
import asyncio
import cv2
import numpy as np
from mss import mss
import pyautogui
from data_annotation import OSUAnnotationTool  # 你的标注工具语音
from knowledge_base import KnowledgeEnhancedChat  
from voice_system import VoiceInteractionSystem  

class CompleteOSUSystem:
    def __init__(self):
        # 初始化所有组件
        self.chat_system = KnowledgeEnhancedChat()
        self.voice_system = VoiceInteractionSystem(self.chat_system)
        self.annotation_tool = OSUAnnotationTool()
        
        # 游戏状态
        self.game_state = {
            'detection_active': False,
            'voice_mode': False,
            'current_difficulty': 'normal',
            'performance_stats': {
                'total_clicks': 0,
                'accuracy': 0.0,
                'max_combo': 0
            }
        }
        
        # 截图设置
        self.monitor = {'top': -30, 'left': 1300, 'width': 1260, 'height': 800}
        self.sct = mss()
        
        # 异步任务
        self.tasks = []
        
    async def start_system(self):
        """启动完整系统"""
        print("=== OSU AI 完整系统启动 ===")
        print("1. 智能聊天系统")
        print("2. 语音交互系统") 
        print("3. 游戏检测系统")
        print("4. 数据标注工具")
        print("输入 'help' 查看可用命令")
        
        # 启动主循环
        await self.main_loop()
    
    async def main_loop(self):
        """主循环"""
        try:
            while True:
                # 检查用户输入
                user_input = await self.check_user_input()
                if user_input:
                    await self.process_user_input(user_input)
                
                # 如果游戏检测激活，执行检测逻辑
                if self.game_state['detection_active']:
                    await self.run_detection_cycle()
                
                # 控制循环频率
                await asyncio.sleep(0.033)  # ~30fps
                
        except KeyboardInterrupt:
            await self.cleanup()
    
    async def process_user_input(self, user_input: str):
        """处理用户输入"""
        if user_input.lower() in ['退出', 'quit']:
            await self.cleanup()
            return
        
        # 特殊命令处理
        if user_input.lower() == '语音模式':
            await self.toggle_voice_mode()
            return
        elif user_input.lower() == '标注模式':
            await self.start_annotation_mode()
            return
        elif user_input.lower() == '帮助' or user_input.lower() == 'help':
            self.show_help()
            return
        
        # 使用聊天系统处理
        response = await self.chat_system.generate_response(user_input, self.game_state)
        print(f"AI: {response['response']}")
        
        # 执行命令
        for command in response['commands']:
            await self.execute_command(command)
    
    async def toggle_voice_mode(self):
        """切换语音模式"""
        if not self.game_state['voice_mode']:
            self.game_state['voice_mode'] = True
            # 在后台启动语音模式
            asyncio.create_task(self.voice_system.start_voice_mode())
            print("语音模式已启动")
        else:
            self.game_state['voice_mode'] = False
            self.voice_system.stop_voice_mode()
            print("语音模式已停止")
    
    async def start_annotation_mode(self):
        """启动标注模式"""
        print("切换到标注模式...")
        # 这里可以集成你的标注工具
        # 注意：这可能会阻塞主线程，需要考虑异步执行
        print("标注功能需要独立运行，请单独启动标注工具")
    
    async def run_detection_cycle(self):
        """运行游戏检测周期"""
        try:
            # 截图
            screenshot = self.sct.grab(self.monitor)
            img = np.array(screenshot)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # 这里可以添加你的检测逻辑
            # 暂时使用简单的显示
            cv2.putText(img_bgr, f"检测模式: {self.game_state['current_difficulty']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_bgr, "按Q退出检测", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('OSU AI System - Detection Mode', img_bgr)
            
            # 非阻塞等待按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.game_state['detection_active'] = False
                cv2.destroyAllWindows()
                
        except Exception as e:
            print(f"检测周期错误: {e}")
    
    async def execute_command(self, command: dict):
        """执行游戏命令"""
        action = command.get('action')
        
        if action == 'start_detection':
            self.game_state['detection_active'] = True
            print("游戏检测已启动")
        elif action == 'stop_detection':
            self.game_state['detection_active'] = False
            cv2.destroyAllWindows()
            print("游戏检测已停止")
        elif action == 'set_difficulty':
            difficulty = command.get('value', 'normal')
            self.game_state['current_difficulty'] = difficulty
            print(f"难度设置为: {difficulty}")
        elif action == 'status_report':
            status = "运行中" if self.game_state['detection_active'] else "已停止"
            print(f"系统状态: 检测{status}, 难度{self.game_state['current_difficulty']}")
    
    async def check_user_input(self):
        """检查用户输入（异步）"""
        # 简化实现，实际可以使用更复杂的异步输入处理
        try:
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(None, input, "> ")
            return user_input.strip()
        except:
            return None
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
=== OSU AI 系统命令 ===
基础命令:
  开始检测 - 启动游戏检测
  停止检测 - 停止游戏检测
  简单难 - 调整难度模式
  状态 - 查看系统状态
  语音模式 - 切换语音控制
  标注模式 - 进入数据标注
  退出 - 退出系统

聊天功能:
  可以询问游戏机制、技巧、术语等
  支持自然语言对话

语音功能:
  支持语音命令控制
  语音反馈系统状态
"""
        print(help_text)
    
    async def cleanup(self):
        """清理资源"""
        print("正在关闭系统...")
        self.game_state['detection_active'] = False
        self.game_state['voice_mode'] = False
        self.voice_system.stop_voice_mode()
        cv2.destroyAllWindows()
        print("系统已安全关闭")

# 运行完整系统
async def main():
    system = CompleteOSUSystem()
    await system.start_system()

if __name__ == "__main__":
    asyncio.run(main())