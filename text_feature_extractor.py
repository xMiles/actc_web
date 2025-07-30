import os
import pandas as pd
import numpy as np
from collections import Counter
from util import *
from constant_variable import *
import re
import torch
import joblib
from transformers import GPT2LMHeadModel, BertTokenizer


class TextFeatureExtractor:
    def __init__(self, full_featrue=False):
        """初始化特征提取器，加载必要的模型"""
        self.full_featrue = full_featrue
        if self.full_featrue:
            self.tfidf_vectorizer = joblib.load(tfidf_model)
        
        self.modern_tokenizer = BertTokenizer.from_pretrained("./models/modern_GPT2")
        self.modern_model = GPT2LMHeadModel.from_pretrained("./models/modern_GPT2")
        self.ancient_tokenizer = BertTokenizer.from_pretrained("./models/tradition_GPT2")
        self.ancient_model = GPT2LMHeadModel.from_pretrained("./models/tradition_GPT2")
        
    def word_token(self, text:str):
        """总字数（单字词）：统计文本中所有的单字个数，不包括标点符号"""
        return len(clean_text(text, 0))

    def word_type(self, text:str):
        """总字符类型数：统计文本中所有的单字类型数，不包括标点符号"""
        return len(set(clean_text(text, 0)))

    def single_word_ratio(self, text:str):
        """单次字占比：求文本中只出现一次的单字占总字符类型数的比例"""
        fre_counter = Counter(clean_text(text, 0))
        single_word = [k for k, v in fre_counter.items() if v == 1]
        return len(single_word) / self.word_type(text)

    def word_MATTR(self, text, moving_window=50):
        """滑动窗口TTR：在每一个窗口里计算里边的TTR值，减少受到文本长度的影响"""
        text = clean_text(text, 0)
        if len(text) < (moving_window + 1):
            ma_ttr = safe_divide(len(set(text)), len(text))
        else:
            sum_ttr = 0
            denorm = 0
            for x in range(len(text)):
                small_text = text[x:(x + moving_window)]
                if len(small_text) < moving_window:
                    break
                denorm += 1
                sum_ttr += safe_divide(len(set(small_text)), float(moving_window))
            ma_ttr = safe_divide(sum_ttr, denorm)
        return ma_ttr

    def word_log_meanFre(self, text):
        """对数平均字频：计算文本中所有单字的对数平均词频，再取对数"""
        text = clean_text(text, 0)
        fre_counter = Counter(text)
        fre_list = [v for k, v in fre_counter.items()]
        return np.log(np.mean(fre_list))

    def word_avg_meaning(self, text):
        """平均义项数：计算文本中所有单字在《汉语大字典》中的平均义项数目"""
        fre_counter = Counter(clean_text(text, 0))
        meaning_sum = 0
        for k in fre_counter:
            if k in classical_chinese_particles:
                meaning_sum += classical_chinese_particles[k]
            elif k in classical_chinese_realwords:
                meaning_sum += classical_chinese_realwords[k]
            elif k in big_meaning_dict:
                meaning_sum += big_meaning_dict[k]
            else:
                meaning_sum += 1
        return meaning_sum / len(fre_counter)

    def common_word_ratio(self, text):
        """常用实词占比：统计文本中所有单字在常用字中的比例"""
        text = clean_text(text, 0)
        fre_counter = Counter(text)
        common_word = [1 for k, v in fre_counter.items() if k in classical_chinese_realwords]
        return sum(common_word) / self.word_type(text)

    def common_praticles_ratio(self, text):
        """常用虚词占比:统计文本中所有单字在常用虚词中的比例"""
        text = clean_text(text, 0)
        fre_counter = Counter(text)
        common_word = [1 for k, v in fre_counter.items() if k in classical_chinese_particles]
        return sum(common_word) / self.word_type(text)

    def highfre_word_ratio(self, text, fre_dict, start, end):
        """高频实词占比：统计文本中所有单字在高频实词中的比例"""
        text = clean_text(text, 0)
        fre_counter = Counter(text)
        common_word = [1 for k, v in fre_counter.items() if k in extract_fre_dict(fre_dict, start, end)]
        return sum(common_word) / self.word_type(text)

    def bigram_type(self, text):
        """双字词类型数：统计文本中所有双字词的类型数"""
        text = clean_text(text, 0)
        bigram_list = [text[i:i+2] for i in range(len(text)-1)]
        return len(set(bigram_list))

    def highfre_bigram_ratio(self, text, fre_dict, start, end):
        """高频虚词占比：统计文本中所有单字在高频虚词中的比例"""
        text = clean_text(text, 0)
        fre_counter = Counter([text[i:i+2] for i in range(len(text)-1)])
        common_word = [1 for k, v in fre_counter.items() if k in extract_fre_dict(fre_dict, start, end)]
        return sum(common_word) / self.bigram_type(text)

    def bigram_mattr(self, text, moving_window=50):
        """bigram的滑动窗口ttr"""
        text = clean_text(text, 0)
        bigram_list = [text[i:i+2] for i in range(len(text)-1)]
        if len(text) < (moving_window + 1):
            ma_ttr = safe_divide(len(set(bigram_list)), len(bigram_list))
        else:
            sum_ttr = 0
            denorm = 0
            for x in range(len(text)):
                small_text = text[x:(x + moving_window)]
                sample_bigram = [small_text[i:i+2] for i in range(len(small_text)-1)]
                if len(small_text) < moving_window:
                    break
                denorm += 1
                sum_ttr += safe_divide(len(set(sample_bigram)), float(len(sample_bigram)))
            ma_ttr = safe_divide(sum_ttr, denorm)
        return ma_ttr
        
    def bigram_log_meanFre(self, text):
        """对数平均双字词频：计算文本中所有bigram的对数平均词频，再取对数"""
        text = clean_text(text, 0)
        bigram_list = [text[i:i+2] for i in range(len(text)-1)]
        fre_counter = Counter(bigram_list)
        fre_list = [v for v in fre_counter.values()]
        return np.log(np.mean(fre_list))

    def sen_avg_len(self, text):
        """平均句长：计算文本中所有句子的平均长度"""
        text = clean_text(text, 1)
        pattern = re.compile(r'[。！？]')
        sen_list = re.split(pattern, text)
        sen_list = [s for s in sen_list if len(s) > 0]  # 过滤空字符串
        if not sen_list:  # 如果没有句子，返回0
            return 0
        sen_len = [len(sen) for sen in sen_list]
        return np.mean(sen_len)

    def sen_savg_len(self, text):
        """平均短句长：计算文本中所有短句的平均长度"""
        text = clean_text(text, 2)
        pattern = re.compile(r'[，。！？]')
        sen_list = re.split(pattern, text)
        sen_list = [s for s in sen_list if len(s) > 0]  # 过滤空字符串
        if not sen_list:  # 如果没有句子，返回0
            return 0
        sen_len = [len(sen) for sen in sen_list]
        return np.mean(sen_len)

    def sen_var(self, text):
        """句长方差：计算文本中所有句子的长度方差"""
        text = clean_text(text, 1)
        pattern = re.compile(r'[。！？]')
        sen_list = re.split(pattern, text)
        sen_list = [s for s in sen_list if len(s) > 0]  # 过滤空字符串
        if not sen_list or len(sen_list) < 2:  # 如果只有0或1个句子，返回0
            return 0
        sen_len = [len(sen) for sen in sen_list]
        return np.var(sen_len)

    def sen_svar(self, text):
        """短句长方差：计算文本中所有短句的长度方差"""
        text = clean_text(text, 2)
        pattern = re.compile(r'[，。！？]')
        sen_list = re.split(pattern, text)
        sen_list = [s for s in sen_list if len(s) > 0]  # 过滤空字符串
        if not sen_list or len(sen_list) < 2:  # 如果只有0或1个句子，返回0
            return 0
        sen_len = [len(sen) for sen in sen_list]
        return np.var(sen_len)

    def common_am_diff(self, text):
        """古今异义个数：统计文本中所有单字在古代汉语与现代汉语中的异义个数"""
        text = clean_text(text, 0)
        word_counter = Counter(text)
        bigram_counter = Counter([text[i:i+2] for i in range(len(text)-1)])
        diff = 0
        for k, v in word_counter.items():
            if k in ancient_modern_words:
                diff += v
        for k, v in bigram_counter.items():
            if k in ancient_modern_words:
                diff += v
        total_count = len(word_counter) + len(bigram_counter)
        return diff / total_count if total_count > 0 else 0
    
    def mt_diff(self, text):
        """ modernity difference 现代性差异"""
        text = clean_text(text, 0)
        ancient_perplexity = self.caculate_paragraph_perplexity(text, self.ancient_tokenizer, self.ancient_model)
        modern_perplexity = self.caculate_paragraph_perplexity(text, self.modern_tokenizer, self.modern_model)

        mt_diff = np.log(safe_divide(modern_perplexity , ancient_perplexity) + 1)
        return mt_diff

    def calculate_perplexity(self, tokenizer, model, sentence):
        """计算单个句子的困惑度"""
        tokenize_input = tokenizer.encode(sentence, 
                                          return_tensors="pt",
                                          truncation=True,  # 截断过长句子
                                          max_length=model.config.max_position_embeddings
                                          )
        with torch.no_grad():  # 不计算梯度，加速计算
            loss = model(tokenize_input, labels=tokenize_input).loss
        return torch.exp(loss).item()

    def caculate_paragraph_perplexity(self, text, tokenizer, model):
        """计算一段文本的困惑度"""
        clean_pattern = re.compile(r'[#\n\t\r ]')
        text = re.sub(clean_pattern, '', text)
        pattern = re.compile(r'[。！？]')
        sen_list = [sen for sen in re.split(pattern, text) if len(sen) > 0]
        if not sen_list:  # 如果没有句子，返回0
            return 0
        perplexity = 0
        for sen in sen_list:
            perplexity += self.calculate_perplexity(tokenizer, model, sen)
        return safe_divide(perplexity, len(sen_list))
    
    def tfidf_vectorize(self, text):
        """使用TF-IDF模型对文本进行向量化"""
        text = [clean_text(text, 0)]
        return self.tfidf_vectorizer.transform(text).toarray()
        
    
    def get_all_features(self, text):
        """提取单段文本的所有特征"""
        features = {}
        
        # 单字特征
        features["word_token"] = self.word_token(text)
        features["single_word_ratio"] = self.single_word_ratio(text)
        features["word_MATTR"] = self.word_MATTR(text)
        features["word_log_meanFre"] = self.word_log_meanFre(text)
        features["word_avg_meaning"] = self.word_avg_meaning(text)
        features["common_word_ratio"] = self.common_word_ratio(text)
        features["common_praticles_ratio"] = self.common_praticles_ratio(text)
        features["common_am_diff"] = self.common_am_diff(text)
        
        # 高频词特征
        features["highfre_word_ratio_500"] = self.highfre_word_ratio(text, word_fre_dict, 0, 500)
        features["highfre_word_ratio_1000"] = self.highfre_word_ratio(text, word_fre_dict, 500, 1000)
        features["highfre_word_ratio_1500"] = self.highfre_word_ratio(text, word_fre_dict, 1000, 1500)
        features["highfre_word_ratio_2000"] = self.highfre_word_ratio(text, word_fre_dict, 1500, 2000)
        
        # 二元组特征
        features["bigram_log_meanFre"] = self.bigram_log_meanFre(text)
        
        # 高频二元组特征
        features["highfre_bigram_ratio_500"] = self.highfre_bigram_ratio(text, bigram_fre_dict, 0, 500)
        features["highfre_bigram_ratio_1000"] = self.highfre_bigram_ratio(text, bigram_fre_dict, 500, 1000)
        features["highfre_bigram_ratio_1500"] = self.highfre_bigram_ratio(text, bigram_fre_dict, 1000, 1500)
        features["highfre_bigram_ratio_2000"] = self.highfre_bigram_ratio(text, bigram_fre_dict, 1500, 2000)
        
        # 句子特征
        features["sen_avg_len"] = self.sen_avg_len(text)
        features["sen_var"] = self.sen_var(text)
        features["sen_svar"] = self.sen_svar(text)

        # 困惑度特征
        features["mt_diff"] = self.mt_diff(text)


        if self.full_featrue:
            # TF-IDF特征
            tfidf_vector = self.tfidf_vectorize(text)
            for i in range(tfidf_vector.shape[1]):
                features[f"tfidf_{i}"] = tfidf_vector[0][i]
        return features
    
    def extract_features_from_texts(self, texts):
        """
        从多个文本中提取特征
        
        参数:
        texts: 可以是单个字符串或字符串列表/Series
        
        返回:
        DataFrame: 包含所有文本特征的数据框
        """
        if isinstance(texts, str):
            # 单个文本
            features = self.get_all_features(texts)
            return pd.DataFrame([features])
        
        # 多个文本
        all_features = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            print(f"处理文本 {i+1}/{total}...")
            features = self.get_all_features(text)
            all_features.append(features)
            
        return pd.DataFrame(all_features)

    def features2vec(self, features, select_features=None):
        """
        将特征转换为向量
        
        参数:
        features: 特征字典
        select_features: 选择的特征列表，如果为None，则使用所有特征
        
        返回:
        list: 特征向量
        """
        # 复制一份
        df = features.copy()

        # 处理BERT以列表形式存储的特征
        list_features = []
        
        # 处理存储值是列表的特征
        for col in df.columns:
            if isinstance(df[col].iloc[0], (list, tuple)) if len(df)>0 else False:
                list_features.append(col)
        
        for col in list_features:
            try:
                list_len = len(df[col].iloc[0])
                for i in range(list_len):
                    df[f"{col}_{i}"] = df[col].apply(lambda x: x[i] if isinstance(x, (list, tuple)) else 0)
                df.drop(columns=[col], inplace=True)
                print(f"处理特征 {col} 成功")
            except Exception as e:
                print(f"处理特征 {col} 时出错: {e}")
    
        if select_features is None:
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

            exclude_features = ['id', 'text', 'id', 'label', 'text_name']
            selected_features = [col for col in numeric_features if col not in exclude_features]
            print(f"选择的特征: {selected_features}")
        
        X = df[selected_features].values

        X = np.nan_to_num(X)  # 将NaN替换为0

        return X

# 使用示例
if __name__ == "__main__":
    import time
    start_time = time.time()
    extractor = TextFeatureExtractor(full_featrue=True)
    
    # 单个文本示例
    sample_text = '一、元年。春王。定何以无正月？正月者，正即位也。定无正月者，即位后也。即位何以后？昭公在外，得入不得入，未可知也。曷为未可知？在季氏也。定、哀多微辞，主人习其读而问其传，则未知己之有罪焉尔。二、三月，晋人执宋仲几于京师。仲几之罪何？不蓑城也。其言于京师何？伯讨也。伯讨则其称人何？贬。曷为贬？不与大夫专执也。曷为不与？实与而文不与。文曷为不与？大夫之义，不得专执也。三、夏六月癸亥，公之丧至自干侯。四、戊辰，公即位。癸亥公之丧至自干侯，则曷为以戊辰之日，然后即位？正棺于两楹之间，然后即位。子沈子曰：“定君乎国，然后即位。”即位不日，此何以日？录乎内也。五、秋七月癸巳，葬我君昭公。六、九月，大雩。七、立炀宫。炀宫者何？炀公之宫也。立者何？立者不宜立也。立炀宫，非礼也。八、冬十月，霣霜杀菽。何以书？记异也。此灾菽也，曷为以异书？异大乎灾也。'
    print("提取单个文本特征:")
    features_df = extractor.extract_features_from_texts(sample_text)
    print(features_df.T)  # 转置后打印，便于查看
    print(f"提取单个文本特征耗时: {time.time() - start_time:.2f} 秒")
    print(features_df.shape)

    sen_len = extractor.sen_avg_len(sample_text)
    print(f"平均句长: {sen_len:.2f}")
    sen_slen = extractor.sen_savg_len(sample_text)
    print(f"平均短句长: {sen_slen:.2f}")
    mt_diff = extractor.mt_diff(sample_text)
    print(f"现代-古代困惑度差异: {mt_diff:.2f}")
    print()
    exit()
    
    # 多个文本示例 - 从文件读取
    print("\n从Excel文件提取多个文本特征:")
    try:
        df = pd.read_excel('./data/uni_text.xlsx')
        # 只处理前3个文本，节省时间
        sample_texts = df['text'].head(3)
        results_df = extractor.extract_features_from_texts(sample_texts)

        # 将原文本和特征合并
        results_df['text'] = df['text']
        results_df['id'] = df['id']

        results_vec = extractor.features2vec(results_df)
        results_ids = np.array(df['id'].tolist())

        # 保存结果
        np.savez('./data/features_with_ids_test.npz', ids=results_ids, vectors=results_vec)
        results_df.to_excel('./data/extracted_features_sample.xlsx', index=False)

        print(f"特征已提取并保存到 './data/extracted_features_sample.xlsx'")

    except Exception as e:
        print(f"从文件提取特征时出错: {str(e)}")