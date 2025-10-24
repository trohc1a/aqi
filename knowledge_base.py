# knowledge_base.py
import json
import os
from typing import Dict, List
from llama_integration import AdvancedLlamaChat

class OSUKnowledgeBase:
    def __init__(self, knowledge_file="osu_knowledge.json"):
        self.knowledge_file = knowledge_file
        self.knowledge = self._load_knowledge()
        
    def _load_knowledge(self) -> Dict:
        """加载游戏知识库"""
        base_knowledge = {
            "game_mechanics": {
                "hit_circles": "需要准确点击的彩色圆圈，点击时机由收缩圈判断",
                "sliders": "需要按住并跟随移动的滑条，包含起点、终点和中间tick点",
                "spinners": "需要快速旋转鼠标的转盘，转满圈数即可",
                "approach_circles": "从外向内收缩的圆圈，当收缩到圆圈边界时点击最佳",
                "combo": "连续成功点击的计数，断连会重置",
                "accuracy": "点击时机准确度的百分比，100%为完美"
            },
            "game_modes": {
                "osu_standard": "标准模式，点击圆圈、滑条和转盘",
                "taiko": "太鼓模式，按照节奏敲击",
                "catch": "接水果模式，移动人物接住掉落的水果", 
                "mania": "下落式音游模式，按照节奏按下对应键位"
            },
            "difficulty_terms": {
                "ar": "Approach Rate，圆圈出现到需要点击的时间",
                "cs": "Circle Size，圆圈的大小",
                "hp": "HP Drain，血条下降速度",
                "od": "Overall Difficulty，判定严格度",
                "stars": "星数，综合难度评级"
            },
            "training_tips": {
                "beginner": "新手建议从低星图开始，专注于准确度而非速度",
                "intermediate": "中级玩家可以练习连打和滑条控制",
                "advanced": "高级玩家需要掌握读图能力和手眼协调",
                "accuracy": "提高准确度的关键是熟悉歌曲节奏和练习目押",
                "speed": "提高速度需要逐渐增加难度，不要急于求成"
            },
            "technical_terms": {
                "pp": "Performance Points，表现分，衡量玩家水平的指标",
                "rank": "排名，基于pp的全球排名",
                "mods": "修改器，可以改变游戏难度的选项",
                "mapper": "谱面制作者",
                "beatmap": "游戏谱面，包含歌曲和点击序列"
            }
        }
        
        # 如果存在自定义知识文件，则合并
        if os.path.exists(self.knowledge_file):
            try:
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    custom_knowledge = json.load(f)
                    base_knowledge.update(custom_knowledge)
            except Exception as e:
                print(f"加载自定义知识库失败: {e}")
        
        return base_knowledge
    
    def query_knowledge(self, question: str) -> List[str]:
        """查询知识库"""
        question_lower = question.lower()
        results = []
        
        # 在各个分类中搜索相关知识点
        for category, contents in self.knowledge.items():
            for term, explanation in contents.items():
                if term in question_lower or any(word in question_lower for word in term.split('_')):
                    results.append(explanation)
        
        return results[:3]  # 返回最相关的3条结果
    
    def add_knowledge(self, category: str, term: str, explanation: str):
        """添加新知识"""
        if category not in self.knowledge:
            self.knowledge[category] = {}
        
        self.knowledge[category][term] = explanation
        self._save_knowledge()
    
    def _save_knowledge(self):
        """保存知识库到文件"""
        try:
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存知识库失败: {e}")

# 知识增强的聊天系统
class KnowledgeEnhancedChat(AdvancedLlamaChat):
    def __init__(self, model_name="llama3.1"):
        super().__init__(model_name)
        self.knowledge_base = OSUKnowledgeBase()
        
    async def generate_response(self, user_input: str, game_context: Dict = None) -> Dict:
        """使用知识库增强的回复生成"""
        # 首先查询知识库
        knowledge_results = self.knowledge_base.query_knowledge(user_input)
        
        # 如果有相关知识，丰富提示词
        if knowledge_results:
            knowledge_context = "\n相关游戏知识:\n" + "\n".join(f"- {item}" for item in knowledge_results)
            user_input = user_input + knowledge_context
        
        # 调用父类方法生成回复
        return await super().generate_response(user_input, game_context)