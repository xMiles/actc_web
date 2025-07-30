import pandas as pd 
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from text_feature_extractor import TextFeatureExtractor
import optuna

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class LightGBMTrainer:
    def __init__(self, selected_features=None):
        """
        初始化 LightGBM 训练器
        
        Args:
            selected_features (list): 选定的特征列表，如果为None则使用所有特征
        """
        self.feature_extractor = TextFeatureExtractor(full_featrue=True)
        self.selected_features = selected_features
        self.model = None
        self.feature_names = None
        
    def load_data(self, data_path='./data/features_v3.xlsx', vector_path='./data/train_vec.npz'):
        """加载数据"""
        print("正在加载数据...")
        
        # 读取原始数据
        df = pd.read_excel(data_path)
        self.df = df
        
        # 加载特征向量
        with np.load(vector_path) as data:
            ids = data['ids']
            vectors = data['vectors']
        
        # 对应标签
        vector_labels = []
        for id_value in ids:
            index = df[df['id'] == id_value].index[0]
            vector_labels.append(df.loc[index, 'label'])
        
        self.vectors = vectors
        self.labels = np.array(vector_labels)
        
        print(f"数据集大小: {vectors.shape}")
        print(f"标签分布: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
        
        return vectors, np.array(vector_labels)
    
    def get_feature_names(self):
        """获取特征名称"""
        if self.feature_names is None:
            # 从特征提取器获取特征名称
            sample_text = self.df['text'].iloc[0]
            features_df = self.feature_extractor.extract_features_from_texts([sample_text])
            self.feature_names = features_df.select_dtypes(include=[np.number]).columns.tolist()
        return self.feature_names
    
    def select_features(self, vectors):
        """选择指定的特征"""
        if self.selected_features is None:
            return vectors
        
        feature_names = self.get_feature_names()
        selected_indices = []
        
        for feature in self.selected_features:
            if feature in feature_names:
                selected_indices.append(feature_names.index(feature))
            else:
                print(f"警告: 特征 '{feature}' 不存在于特征列表中")
        
        if not selected_indices:
            print("警告: 没有找到有效的选定特征，使用所有特征")
            return vectors
        
        print(f"使用 {len(selected_indices)} 个选定特征")
        return vectors[:, selected_indices]
    
    def train_model(self, use_optuna=True, n_trials=100):
        """
        训练 LightGBM 模型
        
        Args:
            use_optuna (bool): 是否使用 Optuna 进行超参数优化
            n_trials (int): Optuna 优化试验次数
        """
        # 选择特征
        X = self.select_features(self.vectors)
        y = self.labels
        
        print(f"训练数据形状: {X.shape}")
        
        if use_optuna:
            # 使用 Optuna 进行超参数优化
            try:
                import optuna
                print("使用 Optuna 进行超参数优化...")
                
                def objective(trial):
                    params = {
                        'objective': 'multiclass',
                        'num_class': len(np.unique(y)),
                        'metric': 'multi_logloss',
                        'boosting_type': 'gbdt',
                        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'verbosity': -1,
                        'random_state': 42
                    }
                    
                    # 使用交叉验证评估
                    cv_scores = cross_val_score(
                        LGBMClassifier(**params), 
                        X, y, 
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=503),
                        scoring='f1_macro'
                    )
                    return cv_scores.mean()
                
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=n_trials)
                
                print("最佳参数:")
                print(study.best_params)
                print(f"最佳交叉验证分数: {study.best_value:.4f}")
                
                # 使用最佳参数训练模型
                best_params = study.best_params
                best_params.update({
                    'objective': 'multiclass',
                    'num_class': len(np.unique(y)),
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'verbosity': -1,
                    'random_state': 42
                })
                
                self.model = LGBMClassifier(**best_params)
                
            except ImportError:
                print("Optuna 未安装，使用默认参数")
                use_optuna = False
        
        if not use_optuna:
            # 使用手动调优的默认参数
            params = {
                'objective': 'multiclass',
                'num_class': len(np.unique(y)),
                'boosting_type': 'gbdt',
                'num_leaves': 100,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'verbosity': -1,
                'random_state': 42,
                'n_estimators': 500
            }
            
            self.model = LGBMClassifier(**params)
        
        # 在完整数据集上训练
        print("在完整数据集上训练模型...")
        self.model.fit(X, y)
        
        # 交叉验证评估
        print("进行交叉验证评估...")
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=503),
            scoring='f1_macro'
        )
        
        print(f"交叉验证 F1-macro 分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.model
    
    def evaluate_model(self, test_size=0.2):
        """评估模型性能"""
        X = self.select_features(self.vectors)
        y = self.labels
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=503, stratify=y
        )
        
        # 在训练集上训练
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # 评估指标
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"测试集准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig('./model/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, report
    
    def plot_feature_importance(self, top_n=20):
        """绘制特征重要性"""
        if self.model is None:
            print("模型未训练，请先训练模型")
            return
        
        feature_names = self.get_feature_names()
        if self.selected_features:
            feature_names = [name for name in feature_names if name in self.selected_features]
        
        importances = self.model.feature_importances_
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"前{top_n}个重要特征:")
        print(importance_df.head(top_n))
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('重要性')
        plt.title(f'前{top_n}个重要特征')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('./model/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def save_model(self, model_path='./model/classifier_lgb.pkl'):
        """保存模型"""
        if self.model is None:
            print("模型未训练，无法保存")
            return
        
        # 创建model目录
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型和相关信息
        model_info = {
            'model': self.model,
            'selected_features': self.selected_features,
            'feature_names': self.get_feature_names()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"模型已保存到: {model_path}")

def main():
    """主训练流程"""
    
    # # 定义选定的特征（根据您的需求修改）
    # selected_features = [
    #     # 基础词汇特征
    #     'word_token',           # 总字数
    #     'word_MATTR',           # 字符多样性
    #     'single_word_ratio',    # 单次出现字比例
    #     'word_log_meanFre',     # 字平均频率
    #     'word_avg_meaning',     # 平均义项数
    #     'common_am_diff',       # 古今异义词比例
    #     'highfre_word_ratio_500', # 高频字比例（500个最常用字）
    #     'highfre_word_ratio_1000', # 高频字比例（1000个最常用字）
    #     'highfre_word_ratio_1500',
    #     'highfre_word_ratio_2000',
    #     'common_word_ratio',    # 常用字比例
    #     'common_praticles_ratio', # 常用虚词比例
    #     'bigram_log_meanFre',   # 二元组平均频率
    #     'highfre_bigram_ratio_500', # 高频二元组比例（500个最常用二元组）
    #     'highfre_bigram_ratio_1000', # 高频二元组比例（1000个最常用二元组）
    #     'highfre_bigram_ratio_1500',
    #     'highfre_bigram_ratio_2000',

    #     # 句法特征
    #     'sen_avg_len',          # 平均句长
    #     'sen_svar',         # 平均短句长
    #     'sen_var',              # 句长变异度
        
    #     # 困惑度
    #     'mt_diff'
    # ]
    
    # print("=== LightGBM 古汉语文本分类模型训练 ===")
    # print(f"选定特征数量: {len(selected_features)}")
    # print("特征列表:", selected_features)
    
    # 初始化训练器
    trainer = LightGBMTrainer()
    
    # 加载数据
    try:
        trainer.load_data()
    except FileNotFoundError as e:
        print(f"数据文件未找到: {e}")
        print("请确保 './data/features_v3.xlsx' 和 './data/train_vec.npz' 文件存在")
        return
    
    # 训练模型
    try:
        trainer.train_model(use_optuna=True, n_trials=50)  # 可以调整试验次数
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("尝试使用默认参数训练...")
        trainer.train_model(use_optuna=False)
    
    # 评估模型
    print("\n=== 模型评估 ===")
    trainer.evaluate_model()
    
    # 特征重要性分析
    print("\n=== 特征重要性分析 ===")
    trainer.plot_feature_importance(top_n=15)
    
    # 保存模型
    trainer.save_model('./model/classifier_lgb.pkl')
    
    print("\n=== 训练完成 ===")
    print("模型文件: './model/classifier_lgb.pkl'")
    print("混淆矩阵: './model/confusion_matrix.png'")
    print("特征重要性图: './model/feature_importance.png'")

if __name__ == "__main__":
    main()
    # model_path = './model/classifier_lgb.pkl'
    # with open(model_path, 'rb') as f:
    #     model = pickle.load(f)
    
    # print("模型加载成功！")
    # print("模型类型:", type(model['model']))
    # print("模型参数:", model['model'].get_params())
