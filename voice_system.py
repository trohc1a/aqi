# voice_system.py
import asyncio
import speech_recognition as sr
import threading
from queue import Queue
import time
from llama_integration import AdvancedLlamaChat

class VoiceInteractionSystem:
    def __init__(self, chat_system):
        self.chat_system = chat_system
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # 语音队列
        self.voice_queue = Queue()
        self.is_listening = False
        self.last_voice_time = 0
        self.voice_cooldown = 2.0  # 语音指令冷却时间
        
        # 设置能量阈值
        self.recognizer.energy_threshold = 500

    async def start_voice_mode(self):
        """启动语音交互模式"""
        print("语音交互模式已启动...")
        self.is_listening = True
        
        # 在后台线程中运行语音监听
        voice_thread = threading.Thread(target=self._voice_listener, daemon=True)
        voice_thread.start()
        
        # 在主线程中处理语音指令
        await self._process_voice_commands()

    def _voice_listener(self):
        """在后台线程中监听语音输入"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("麦克风已校准，正在聆听...")
            
            # 添加调试：显示当前灵敏度
            print(f"当前能量阈值: {self.recognizer.energy_threshold}")
            print("提示：如果一直检测不到，尝试调低 energy_threshold")
            
            while self.is_listening:
                try:
                    # 监听语音输入
                    print("等待语音中...")  # 添加这行看是否在循环
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    print("检测到语音！正在识别...")  # 如果能到这里说明听到声音了
                    
                    # 语音识别
                    text = self.recognizer.recognize_google(audio, language='zh-CN')
                    print(f"识别到语音: {text}")
                    
                    # 添加到处理队列
                    self.voice_queue.put(text)
                    
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    print("听到声音但无法识别内容")
                except Exception as e:
                    print(f"语音识别错误: {e}")

    async def _process_voice_commands(self):
        """处理语音指令队列"""
        while self.is_listening:
            try:
                # 非阻塞获取语音指令
                if not self.voice_queue.empty():
                    voice_text = self.voice_queue.get_nowait()
                    current_time = time.time()
                    
                    # 检查冷却时间
                    if current_time - self.last_voice_time >= self.voice_cooldown:
                        self.last_voice_time = current_time
                        
                        print(f"处理语音指令: {voice_text}")
                        
                        # 使用聊天系统处理指令
                        response = await self.chat_system.generate_response(voice_text)
                        
                        # 语音回复
                        self.speak(response['response'])
                        
                        # 执行命令
                        if response['commands']:
                            print(f"执行命令: {response['commands']}")
                            
            except Exception as e:
                print(f"语音处理错误: {e}")
            
            await asyncio.sleep(0.1)

    def speak(self, text: str):
        """文本转语音 - 需要替换为 VITS 实现"""
        print(f"[TTS] 需要合成语音: {text}")
        # TODO: 在这里添加 VITS 语音合成代码
        # 目前只是打印文本，您需要替换为实际的 VITS 调用
        
    def test_microphone(self) -> bool:
        """测试麦克风功能"""
        try:
            print("开始麦克风测试...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("请说话...")
                
                audio = self.recognizer.listen(
                    source, 
                    timeout=5, 
                    phrase_time_limit=3
                )
                
                text = self.recognizer.recognize_google(audio, language='zh-CN')
                print(f"测试成功！识别结果: {text}")
                return True
                
        except sr.WaitTimeoutError:
            print("测试失败：超时时间内未检测到语音")
            return False
        except Exception as e:
            print(f"麦克风测试失败: {e}")
            return False

    def test_vits(self, text="你好，这是一个语音合成测试"):
        """测试 VITS 语音合成功能"""
        print(f"测试 VITS 合成: {text}")
        self.speak(text)
        return True

    def stop_voice_mode(self):
        """停止语音交互"""
        self.is_listening = False
        print("语音交互已停止")

# 语音命令快捷方式
class VoiceShortcuts:
    def __init__(self):
        self.shortcuts = {
            '开始游戏': 'start_detection',
            '停止游戏': 'stop_detection', 
            '简单模式': 'set_difficulty easy',
            '困难模式': 'set_difficulty hard',
            '状态报告': 'status_report',
            '帮助': 'help'
        }
    
    def get_shortcut_command(self, voice_text: str):
        """获取语音快捷命令"""
        for phrase, command in self.shortcuts.items():
            if phrase in voice_text:
                return command
        return None

# 示例使用代码
async def main():
    # 初始化聊天系统
    chat_system = AdvancedLlamaChat()
    
    # 初始化语音系统
    voice_system = VoiceInteractionSystem(chat_system)
    
    # 测试麦克风
    if voice_system.test_microphone():
        print("麦克风测试成功")
    else:
        print("麦克风测试失败，请检查设备")
        return
    
    # 测试语音合成
    voice_system.test_vits("你好，这是一个语音合成测试")
    
    # 启动语音交互模式
    await voice_system.start_voice_mode()

# 运行主程序
if __name__ == "__main__":
    asyncio.run(main())