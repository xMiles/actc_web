# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import pickle
import os
from io import BytesIO
import base64

from text_feature_extractor import TextFeatureExtractor

class TextLevelAnalyzer:
    """
    文本层次分析器
    将特征按词、句、篇章三个层面分类，并基于分位数进行评分
    """
    
    def __init__(self, reference_data_path='./data/features_v3.xlsx'):
        """
        初始化分析器
        
        Args:
            reference_data_path: 参考数据文件路径，用于计算分位数
        """
        self.reference_data_path = reference_data_path
        self.reference_stats = None
        
        # 定义特征层次分类
        self.feature_levels = {
            'word_level': {  # 词层面
                'features': [
                    'word_token',           # 总字数
                    'word_log_meanFre',     # 字符平均频率对数
                    'word_avg_meaning',     # 平均字义数
                    'highfre_word_ratio_500', # 高频词比例
                    'highfre_word_ratio_2000', # 高频词比例
                    'highfre_bigram_ratio_500', # 高频二元组比例
                    'highfre_bigram_ratio_2000', # 高频二元组比例
                    'bigram_log_meanFre',   # 二元组平均频率对数
                    'common_word_ratio',    # 常用实词比例
                    'common_praticles_ratio', # 常用虚词比例
                    'common_am_diff',       # 古今常用字差异
                ],
                'name': '字词常用度',
                'description': '反映文章中字词的常用度'
            },
            'word_variety': {
                'features': [
                    'single_word_ratio',    # 单次出现字比例
                    'word_MATTR',           # 字符多样性
                ],
                'name': '词汇多样性',
                'description': '反映文章中词汇的多样性'
            },
            'sentence_level': {  # 句层面
                'features': [
                    'sen_avg_len',          # 平均句长
                    'sen_var',              # 句长变异度
                    'sen_svar',             # 短句长变异度
                ],
                'name': '句式复杂度',
                'description': '反映文章句式结构的复杂程度'
            },
            'discourse_level': {  # 篇章层面
                'features': [
                    'mt_diff',              # 古今困惑度差异
                    
                ],
                'name': '语体典雅度',
                'description': '反映文本的语篇风格与现代汉语的差异程度'
            }
        }
        
        # 特征评分方向（True表示值越大分数越高，False表示值越小分数越高）
        self.feature_directions = {
            'word_token': True,            # 字数多 = 复杂度高
            'single_word_ratio': False,    # 单次字多 = 复杂度高
            'word_MATTR': False,            # 多样性高 = 复杂度高
            'word_log_meanFre': True,     # 频率低 = 复杂度高
            'word_avg_meaning': False,      # 字义多 = 复杂度高
            'common_word_ratio': False,    # 常用字多 = 复杂度低
            'common_praticles_ratio': False, # 虚词多 = 句法复杂
            'bigram_log_meanFre': True,   # 二元组频率低 = 复杂度高
            'sen_avg_len': True,           # 句子长 = 复杂度高
            'sen_var': True,               # 句长变化大 = 复杂度高
            'sen_svar': True,              # 短句变化大 = 复杂度高
            'mt_diff': True,               # 差异大 = 时代特征明显
            'common_am_diff': True,        # 古今差异大 = 古典性强
            'highfre_word_ratio_500': False, # 高频词比例高 = 复杂度低
            'highfre_word_ratio_2000': True, # 高频词比例高 = 复杂度高
            'highfre_bigram_ratio_500': False, # 高频二元组比例高 = 复杂度低
            'highfre_bigram_ratio_2000': False, # 高频二元组比例高 = 复杂度高
        }
        
        self.feature_name_mapping = {
            'word_token': '总字数',            # 字数多 = 复杂度高
            'single_word_ratio': '单次字比例',    # 单次字多 = 复杂度高
            'word_MATTR': '字MATTR',            # 多样性高 = 复杂度高
            'word_log_meanFre': '古籍对数平均字频',     # 频率低 = 复杂度高
            'word_avg_meaning': '字平均义项数',      # 字义多 = 复杂度高
            'common_word_ratio': '常用实词比例',    # 常用字多 = 复杂度低
            'common_praticles_ratio': '常用虚词比例', # 虚词多 = 句法复杂
            'bigram_log_meanFre': '古籍对数平均bigram频率',   # 二元组频率低 = 复杂度高
            'sen_avg_len': '平均句长',           # 句子长 = 复杂度高
            'sen_var': '句长破碎度',               # 句长变化大 = 复杂度高
            'sen_svar': '短句长破碎度',              # 短句变化大 = 复杂度高
            'mt_diff': '语体典雅度',               # 差异大 = 时代特征明显
            'common_am_diff': '常见古今异义词频率',        # 古今差异大 = 古典性强
            'highfre_word_ratio_500': '高频词比例500', # 高频词比例高 = 复杂度低
            'highfre_word_ratio_2000': '高频词比例2000', # 高频词比例高 = 复杂度高
            'highfre_bigram_ratio_500': '高频二元组比例500', # 高频二元组比例高 = 复杂度低
            'highfre_bigram_ratio_2000': '高频二元组比例2000', # 高频二元组比例高 = 复杂度高
        }
        # 加载参考数据
        self._load_reference_data()
    
    def _load_reference_data(self):
        """加载参考数据并计算统计信息"""
        try:
            if os.path.exists(self.reference_data_path):
                print(f"加载参考数据: {self.reference_data_path}")
                df = pd.read_excel(self.reference_data_path)
                
                # 从text_feature_extractor提取特征（如果需要）
                # 这里假设Excel文件已经包含了特征列
                
                # 计算所有特征的分位数统计
                self.reference_stats = {}
                
                # 获取所有需要的特征
                all_features = []
                for level_info in self.feature_levels.values():
                    all_features.extend(level_info['features'])
                
                # 计算每个特征的分位数
                for feature in all_features:
                    if feature in df.columns:
                        values = df[feature].dropna()
                        if len(values) > 0:
                            self.reference_stats[feature] = {
                                'min': values.min(),
                                'max': values.max(),
                                'mean': values.mean(),
                                'std': values.std(),
                                'percentiles': {
                                    i: values.quantile(i/100) for i in range(0, 101, 5)
                                }
                            }
                
                print(f"成功加载 {len(self.reference_stats)} 个特征的统计信息")
            else:
                print(f"参考数据文件不存在: {self.reference_data_path}")
                self._create_dummy_stats()
                
        except Exception as e:
            print(f"加载参考数据失败: {e}")
            self._create_dummy_stats()
    
    def _create_dummy_stats(self):
        """创建虚拟统计数据用于测试"""
        print("创建虚拟统计数据...")
        self.reference_stats = {}
        
        # 为每个特征创建合理的虚拟分布
        dummy_ranges = {
            'word_token': (50, 500),
            'word_type': (30, 300),
            'single_word_ratio': (0.3, 0.8),
            'word_MATTR': (0.5, 0.95),
            'word_log_meanFre': (2, 8),
            'word_avg_meaning': (1, 5),
            'common_word_ratio': (0.2, 0.7),
            'common_praticles_ratio': (0.05, 0.3),
            'bigram_type': (20, 200),
            'bigram_mattr': (0.6, 0.98),
            'bigram_log_meanFre': (1, 6),
            'sen_avg_len': (8, 25),
            'sen_savg_len': (6, 20),
            'sen_var': (5, 50),
            'sen_svar': (3, 30),
            'modern_perplexity': (10, 100),
            'tradition_perplexity': (5, 80),
            'mt_diff': (0, 3),
            'common_am_diff': (0, 1),
        }
        
        for feature, (min_val, max_val) in dummy_ranges.items():
            # 创建正态分布的虚拟数据
            mean_val = (min_val + max_val) / 2
            std_val = (max_val - min_val) / 6
            
            # 生成虚拟数据点
            dummy_values = np.random.normal(mean_val, std_val, 1000)
            dummy_values = np.clip(dummy_values, min_val, max_val)
            
            self.reference_stats[feature] = {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val,
                'percentiles': {
                    i: np.percentile(dummy_values, i) for i in range(0, 101, 5)
                }
            }
    
    def calculate_feature_score(self, feature_name: str, feature_value: float) -> float:
        """
        基于分位数计算单个特征的分数 (0-10分)
        
        Args:
            feature_name: 特征名称
            feature_value: 特征值
            
        Returns:
            0-10分的分数
        """
        if feature_name not in self.reference_stats:
            return 5.0  # 默认中等分数
        
        stats = self.reference_stats[feature_name]
        percentiles = stats['percentiles']
        
        # 找到该值在分位数中的位置
        percentile_rank = 50  # 默认50%分位
        
        for p in range(0, 101, 5):
            if feature_value <= percentiles[p]:
                if p > 0:
                    # 线性插值
                    lower_p = p - 5
                    lower_val = percentiles[lower_p]
                    upper_val = percentiles[p]
                    
                    if upper_val != lower_val:
                        interpolated = lower_p + 5 * (feature_value - lower_val) / (upper_val - lower_val)
                        percentile_rank = interpolated
                    else:
                        percentile_rank = p
                else:
                    percentile_rank = 0
                break
        else:
            percentile_rank = 100
        
        # 根据特征方向调整分数
        if self.feature_directions.get(feature_name, True):
            # 值越大分数越高
            score = percentile_rank / 10
        else:
            # 值越小分数越高
            score = (100 - percentile_rank) / 10
        
        return max(0, min(10, score))
    
    def analyze_text_levels(self, feature_values: Dict[str, float]) -> Dict[str, Any]:
        """
        分析文本的四个层次并评分
        
        Args:
            feature_values: 特征值字典
            
        Returns:
            包含各层次分析结果的字典
        """
        results = {}
        
        for level_key, level_info in self.feature_levels.items():
            level_scores = {}
            level_features = {}
            
            # 计算该层次每个特征的分数
            for feature in level_info['features']:
                if feature in feature_values:
                    value = feature_values[feature]
                    score = self.calculate_feature_score(feature, value)
                    level_scores[feature] = score
                    level_features[feature] = value
            
            # 计算层次平均分
            if level_scores:
                avg_score = np.mean(list(level_scores.values()))
            else:
                avg_score = 5.0
            
            results[level_key] = {
                'name': level_info['name'],
                'description': level_info['description'],
                'avg_score': avg_score,
                'feature_scores': level_scores,
                'feature_values': level_features,
                'max_score': max(level_scores.values()) if level_scores else 5.0,
                'min_score': min(level_scores.values()) if level_scores else 5.0,
            }
        
        return results
    
    def create_level_chart(self, analysis_results: Dict[str, Any], save_path: str = None) -> str:
        """
        创建四层次评分柱状图（简洁版本）
        
        Args:
            analysis_results: 分析结果
            save_path: 保存路径
            
        Returns:
            图表的base64编码字符串
        """
        sns.set_style("ticks")

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
        
        # 左图：四层次总分对比（保持不变）
        levels = []
        scores = []
        colors = []
        color_map = {'word_level': '#FF6B6B', 'sentence_level': '#4ECDC4', 'discourse_level': '#45B7D1', "word_variety": "#C940E4"}
        
        for i, (level_key, result) in enumerate(analysis_results.items()):
            levels.append(result['name'])
            scores.append(result['avg_score'])
            color = color_map.get(level_key, '#FF6B6B')
            colors.append(color)
        
        bars1 = ax1.bar(levels, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_title('文本复杂度四层次评分', fontsize=14, fontweight='bold')
        ax1.set_ylabel('评分 (0-10分)', fontsize=12)
        ax1.set_ylim(0, 10)
        ax1.grid(axis='y', alpha=0.3)
        
        # 在柱子上显示分数
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 添加评分等级线
        score_levels = [2, 4, 6, 8]
        level_names = ['较低', '中等', '较难', '很难']
        for score, name in zip(score_levels, level_names):
            ax1.axhline(y=score, color='gray', linestyle='--', alpha=0.5)
            ax1.text(len(levels)-0.5, score+0.1, name, fontsize=9, alpha=0.7)
        
        # # 右图：按组和分数排序的特征
        # all_features = []
        
        # # 定义层次顺序
        # level_order = ['word_level', 'word_variety', 'sentence_level', 'discourse_level']
        
        # # 按定义的顺序遍历层次
        # for level_key in level_order:
        #     if level_key in analysis_results:
        #         result = analysis_results[level_key]
                
        #         # 获取该层次的特征，按分数排序
        #         level_features = []
        #         for feature, score in result['feature_scores'].items():
        #             chinese_name = self.feature_name_mapping.get(feature, feature)
        #             level_features.append({
        #                 'chinese_name': chinese_name,
        #                 'score': score,
        #                 'color': color_map.get(level_key, '#999999')
        #             })
                
        #         # 按分数从高到低排序
        #         level_features.sort(key=lambda x: x['score'], reverse=True)
        #         all_features.extend(level_features)
        
        # if all_features:
        #     feature_names = [item['chinese_name'] for item in all_features]
        #     feature_scores = [item['score'] for item in all_features]
        #     feature_colors = [item['color'] for item in all_features]
            
        #     # 绘制水平柱状图
        #     bars2 = ax2.barh(range(len(feature_names)), feature_scores, 
        #                     color=feature_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            
        #     # 设置y轴
        #     ax2.set_yticks(range(len(feature_names)))
        #     ax2.set_yticklabels(feature_names, fontsize=10)
            
        #     # 在柱子上显示分数
        #     for i, (bar, score) in enumerate(zip(bars2, feature_scores)):
        #         width = bar.get_width()
        #         ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
        #                 f'{score:.1f}', ha='left', va='center', fontsize=9, fontweight='bold')
            
        #     # 设置x轴和标题
        #     ax2.set_xlabel('评分 (0-10分)', fontsize=12)
        #     ax2.set_title('各特征详细评分（按组分类）', fontsize=14, fontweight='bold')
        #     ax2.set_xlim(0, 10)
        #     ax2.grid(axis='x', alpha=0.3)
            
        #     # 反转y轴，让第一组在上面
        #     ax2.invert_yaxis()
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_str
    
    def generate_text_report(self, analysis_results: Dict[str, Any], text_sample: str = "") -> str:
        """
        生成文本分析报告
        
        Args:
            analysis_results: 分析结果
            text_sample: 文本样本（可选）
            
        Returns:
            HTML格式的报告
        """
        html_report = f"""
        <div class="text-analysis-report">
            <h3>文本复杂度分析报告</h3>
            
            {f'<div class="text-sample"><strong>文本样本：</strong>{text_sample[:100]}...</div>' if text_sample else ''}
            
            <div class="level-summaries">
        """
        
        for level_key, result in analysis_results.items():
            score = result['avg_score']
            
            # 确定评级
            if score >= 8:
                grade = "很高"
                grade_color = "#ED4928"
            elif score >= 6:
                grade = "较高"
                grade_color = "#F36C12"
            elif score >= 4:
                grade = "中等"
                grade_color = "#FFD323"
            else:
                grade = "较低"
                grade_color = "#27AE60"
            
            html_report += f"""
                <div class="level-summary">
                    <h4>{result['name']} <span style="color: {grade_color};">({score:.1f}分 - {grade})</span></h4>
                    <p>{result['description']}</p>
                    
                    <div class="feature-details">
                        <strong>特征详情：</strong>
                        <ul>
            """
            
            for feature, feature_score in result['feature_scores'].items():
                feature_value = result['feature_values'].get(feature, 'N/A')
                html_report += f"<li>{feature}: {feature_value} (评分: {feature_score:.1f})</li>"
            
            html_report += """
                        </ul>
                    </div>
                </div>
            """
        
        html_report += """
            </div>
        </div>
        """
        
        return html_report

# 使用示例和测试函数
def main():
    """测试分析器"""
    # 创建分析器实例
    analyzer = TextLevelAnalyzer()
    feature_extractor = TextFeatureExtractor()
    text = "近腊月下，景气和畅，故山殊可过。足下方温经，猥不敢相烦，辄便往山中，憩感配寺，与山僧饭讫而去。北涉玄灞，清月映郭。夜登华子冈，辋水沦涟，与月上下。寒山远火，明灭林外。深巷寒犬，吠声如豹。村墟夜舂，复与疏钟相间。此时独坐，僮仆静默，多思曩昔，携手赋诗，步仄径，临清流也。当待春中，草木蔓发，春山可望，轻鲦出水，白鸥矫翼，露湿青皋，麦陇朝雊，斯之不远，倘能从我游乎？非子天机清妙者，岂能以此不急之务相邀。然是中有深趣矣！无忽。因驮黄檗人往，不一，山中人王维白。"
    # 模拟特征值
    features_df = feature_extractor.extract_features_from_texts(text)
    test_features = features_df.iloc[0].to_dict()
    
    # 进行分析
    results = analyzer.analyze_text_levels(test_features)
    
    # 打印结果
    print("=== 文本层次分析结果 ===")
    for level_key, result in results.items():
        print(f"\n{result['name']}: {result['avg_score']:.1f}分")
        print(f"描述: {result['description']}")
        print("特征分数:")
        for feature, score in result['feature_scores'].items():
            print(f"  {feature}: {score:.1f}")
    
    # 生成图表
    chart_b64 = analyzer.create_level_chart(results, './text_level_analysis.png')
    print(f"\n图表已生成，base64长度: {len(chart_b64)}")
    
    # 生成报告
    report = analyzer.generate_text_report(results, "这是一段测试文本...")
    print(f"\n报告已生成，长度: {len(report)}")
    
    return analyzer, results, chart_b64, report

if __name__ == "__main__":
    main()