import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import torch
from util import clean_text

# 导入自定义特征提取器和异常检测器
from text_feature_extractor import TextFeatureExtractor
from text_anomaly_detector import TextAnomalyDetector
from text_level_analyzer import TextLevelAnalyzer

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

app = Flask(__name__)

# 加载模型和初始化组件
model = None
extractor = None
anomaly_detector = None
level_analyzer = None

def load_model():
    """加载模型和初始化组件"""
    global model, extractor, anomaly_detector, level_analyzer
    
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'classifier_lgb.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            file = pickle.load(f)
        # 保存时model键存储模型
        model = file['model']
        print("模型加载成功！")
        
        # 初始化特征提取器
        extractor = TextFeatureExtractor(full_featrue=True)
        print("特征提取器初始化成功！")
        
        # 初始化异常检测器
        anomaly_detector = TextAnomalyDetector()
        print("异常检测器初始化成功！")
        
        # 初始化层次分析器
        level_analyzer = TextLevelAnalyzer()
        print("层次分析器初始化成功！")
        
        return True
        
    except Exception as e:
        print(f"模型加载或初始化失败: {e}")
        model = None
        extractor = None
        anomaly_detector = None
        level_analyzer = None
        return False

# 在应用启动时加载模型
load_model()

# 类别映射（根据您的实际分类调整）
categories = {
    0: "初级",
    1: "中级", 
    2: "高级",
}

# 在文件顶部添加特征名称映射
feature_name_mapping = {
    'word_token': '总字数',
    'word_log_meanFre': '字符平均频率对数',
    'word_avg_meaning': '平均义项数',
    'highfre_word_ratio_500': '高频词比例(前500)',
    'highfre_word_ratio_1000': '高频词比例(前1000)',
    'highfre_word_ratio_1500':'高频词比例(1500)',
    'highfre_word_ratio_2000': '高频词比例(前2000)',
    'highfre_bigram_ratio_500': '高频二元组比例(前500)',
    'highfre_bigram_ratio_1000': '高频二元组比例(前1000)',
    'highfre_bigram_ratio_1500': '高频二元组比例(前1500)',
    'highfre_bigram_ratio_2000': '高频二元组比例(前2000)',
    'bigram_log_meanFre': '二元组平均频率对数',
    'common_word_ratio': '常用实词比例',
    'common_praticles_ratio': '常用虚词比例',
    'common_am_diff': '常用古今异义词比例',
    'single_word_ratio': '单次字比例',
    'word_MATTR': '字符多样性指数',
    'sen_avg_len': '平均句长',
    'sen_var': '句长变异度',
    'sen_svar': '短句长变异度',
    'mt_diff': '语体典雅度',

}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if model is None or extractor is None or anomaly_detector is None:
        return jsonify({'error': '模型或特征提取器加载失败，请检查模型文件'})
    
    text = request.form.get('text', '')
    text = clean_text(text, mode=3)
    
    # 验证输入
    if not text:
        return jsonify({'error': '请输入文本'})
    
    # 检查字符数限制
    if len(text) > 3000:
        return jsonify({'error': '文本长度超过3000字，请减少内容'})
    
    try:
        # 异常检测
        anomaly_results = anomaly_detector.detect_anomalies(text)
        
        # 如果有严重错误，阻止分类
        if anomaly_detector.has_blocking_errors(text):
            error_messages = [error['message'] for error in anomaly_results['errors']]
            return jsonify({
                'error': '文本不符合古汉语分析要求',
                'error_type': 'text_quality',
                'error_details': error_messages,
                'suggestions': list(set(anomaly_results['suggestions']))  # 去重
            })
        
        # 提取特征
        features_df = extractor.extract_features_from_texts(text)
        
        # 获取特征向量
        features_vec = extractor.features2vec(features_df)
        
        # 预测分类
        prediction = model.predict(features_vec)[0]
        category = categories.get(prediction, "未知类别")
        
        # 获取概率分布（如果模型支持）
        prob_data = {}
        if hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')):
            probabilities = model.predict_proba(features_vec)[0]
            prob_data = {categories.get(i, f"类别{i}"): float(prob) 
                         for i, prob in enumerate(probabilities)}
        
        # 获取特征值信息
        feature_values = {}
        for col in features_df.columns:
            value = features_df[col].iloc[0]
            if isinstance(value, (int, float)):
                # 统一保留2位小数
                if abs(value) < 0.001:
                    feature_values[col] = f"{value:.2e}"  # 科学计数法，2位小数
                else:
                    feature_values[col] = f"{value:.2f}"   # 普通格式，2位小数
            else:
                feature_values[col] = str(value)
        
        
        # 层次分析
        level_analysis = None
        level_chart = None
        
        if level_analyzer is not None:
            try:
                # 将特征值转换为浮点数字典
                numeric_feature_values = {}
                for col in features_df.columns:
                    try:
                        numeric_feature_values[col] = float(features_df[col].iloc[0])
                    except (ValueError, TypeError):
                        continue
                
                # 进行层次分析
                level_analysis = level_analyzer.analyze_text_levels(numeric_feature_values)
                
                # 生成层次分析图表
                level_chart = level_analyzer.create_level_chart(level_analysis)
                
                print("层次分析完成")
                
            except Exception as e:
                print(f"层次分析失败: {e}")
        
        
        
        # 准备返回数据
        response_data = {
            'category': category,
            'probabilities': prob_data,
            # 新增层次分析数据
            'level_analysis': level_analysis,
            'level_chart': level_chart,
        }
        
        # 如果有警告，添加警告信息
        if anomaly_results['warnings']:
            response_data['warnings'] = [warning['message'] for warning in anomaly_results['warnings']]
            response_data['warning_suggestions'] = [warning['suggestion'] for warning in anomaly_results['warnings']]
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'分类过程发生错误: {str(e)}'})

@app.route('/model_status')
def model_status():
    """检查模型状态的API端点"""
    status = {
        'model_loaded': model is not None,
        'extractor_loaded': extractor is not None,
        'anomaly_detector_loaded': anomaly_detector is not None,
        'level_analyzer_loaded': level_analyzer is not None,
        'model_type': type(model).__name__ if model else None
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True)