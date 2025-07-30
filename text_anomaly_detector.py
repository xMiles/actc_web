import re
from typing import Dict, List, Any

class TextAnomalyDetector:
    """
    文本异常监测类（简化版）
    用于检测古汉语文本中的基本异常情况
    """
    
    def __init__(self):
        # 定义异常检测的阈值
        self.thresholds = {
            'min_punctuation_ratio': 0,  # 最小标点符号比例
            'max_punctuation_ratio': 0.4,    # 最大标点符号比例
            'min_text_length': 30,            # 最小文本长度
            'max_number_ratio': 0,         # 最大数字比例
            'max_english_ratio': 0.05,       # 最大英文字符比例
        }
        
        # 常见古汉语标点符号
        self.chinese_punctuation = '，。！？；：""''（）【】《》、…—'
        
        # 常见数字字符（只包含阿拉伯数字）
        self.number_chars = '0123456789'
        
        # 英文字符
        self.english_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
    def detect_anomalies(self, text: str) -> Dict[str, Any]:
        """
        检测文本中的异常
        
        Args:
            text (str): 待检测的文本
            
        Returns:
            Dict[str, Any]: 包含异常检测结果的字典
        """
        results = {
            'has_anomalies': False,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # 基础统计信息
        stats = self._calculate_basic_stats(text)
        
        # 执行异常检测
        anomaly_checks = [
            self._check_text_length,
            self._check_punctuation_ratio,
            self._check_number_ratio,
            self._check_english_ratio,
        ]
        
        for check_func in anomaly_checks:
            anomaly = check_func(text, stats)
            if anomaly:
                if anomaly['severity'] == 'error':
                    results['errors'].append(anomaly)
                    results['suggestions'].append(anomaly['suggestion'])
                else:
                    results['warnings'].append(anomaly)
                    results['suggestions'].append(anomaly['suggestion'])
        
        # 判断是否有异常
        results['has_anomalies'] = len(results['errors']) > 0 or len(results['warnings']) > 0
        
        # 如果没有任何问题，添加成功提示
        if not results['has_anomalies']:
            results['suggestions'].append("文本符合古汉语分析要求")
            
        return results
    
    def _calculate_basic_stats(self, text: str) -> Dict[str, Any]:
        """计算文本基础统计信息"""
        stats = {}
        
        # 基本长度信息
        stats['total_length'] = len(text)
        
        # 标点符号统计
        punct_count = len([c for c in text if c in self.chinese_punctuation])
        stats['punctuation_ratio'] = punct_count / max(1, stats['total_length'])
        
        # 数字统计
        number_count = len([c for c in text if c in self.number_chars])
        stats['number_ratio'] = number_count / max(1, stats['total_length'])
        
        # 英文字符统计
        english_count = len([c for c in text if c in self.english_chars])
        stats['english_ratio'] = english_count / max(1, stats['total_length'])
        
        return stats
    
    def _check_text_length(self, text: str, stats: Dict) -> Dict[str, Any]:
        """检测文本长度异常"""
        if stats['total_length'] < self.thresholds['min_text_length']:
            return {
                'type': 'text_length',
                'severity': 'warning',
                'message': f"文本过短（{stats['total_length']}字），无法进行准确分类",
                'suggestion': f"至少{self.thresholds['min_text_length']}字的古汉语文本能够达到较为准确的分类结果"
            }
        return None
    
    def _check_punctuation_ratio(self, text: str, stats: Dict) -> Dict[str, Any]:
        """检测标点符号比例异常"""
        if stats['punctuation_ratio'] > self.thresholds['max_punctuation_ratio']:
            return {
                'type': 'punctuation_ratio',
                'severity': 'warning',
                'message': f"标点符号过多（占比{stats['punctuation_ratio']:.1%}）",
                'suggestion': "建议适当减少标点符号，保持文本的连贯性"
            }
        elif stats['punctuation_ratio'] < self.thresholds['min_punctuation_ratio']:
            return {
                'type': 'punctuation_ratio',
                'severity': 'warning',
                'message': f"标点符号过少（占比{stats['punctuation_ratio']:.1%}）",
                'suggestion': "建议合理句读后再使用应用进行分类"
            }
        return None
    
    def _check_number_ratio(self, text: str, stats: Dict) -> Dict[str, Any]:
        """检测数字比例异常"""
        if stats['number_ratio'] > self.thresholds['max_number_ratio']:
            return {
                'type': 'number_ratio',
                'severity': 'error',
                'message': f"数字字符过多（占比{stats['number_ratio']:.1%}）",
                'suggestion': "古汉语文本中不应包含过多阿拉伯数字，请检查文本内容"
            }
        return None
    
    def _check_english_ratio(self, text: str, stats: Dict) -> Dict[str, Any]:
        """检测英文字符比例异常"""
        if stats['english_ratio'] > self.thresholds['max_english_ratio']:
            return {
                'type': 'english_ratio',
                'severity': 'error',
                'message': f"包含英文字符（占比{stats['english_ratio']:.1%}）",
                'suggestion': "请移除所有英文字符，仅使用古汉语文本"
            }
        return None
    
    def has_blocking_errors(self, text: str) -> bool:
        """
        检查是否有阻止分类的严重错误
        
        Args:
            text (str): 待检测的文本
            
        Returns:
            bool: True表示有严重错误，应阻止分类；False表示可以继续分类
        """
        results = self.detect_anomalies(text)
        return len(results['errors']) > 0

# 测试函数
if __name__ == "__main__":
    detector = TextAnomalyDetector()
    
    # 测试文本
    test_texts = [
        "子曰：学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？",  # 正常文本
        "aaa bbb ccc ddd eee fff",                                    # 英文文本
        "。。。。。。。。。。。。。。。。。。。。。。。。。",              # 重复字符
        "123456789",                                                   # 数字文本
        "短文本",  
        "子曰学而时习之不亦悦乎有朋自远方来不亦乐乎"                                                    # 过短文本
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n=== 测试文本 {i+1} ===")
        print(f"文本: {text}")
        
        results = detector.detect_anomalies(text)
        print(f"有异常: {results['has_anomalies']}")
        print(f"阻止分类: {detector.has_blocking_errors(text)}")
        
        if results['errors']:
            print("错误:")
            for error in results['errors']:
                print(f"  - {error['message']}")
                
        if results['warnings']:
            print("警告:")
            for warning in results['warnings']:
                print(f"  - {warning['message']}")
                
        if results['suggestions']:
            print("建议:")
            for suggestion in results['suggestions']:
                print(f"  - {suggestion}")