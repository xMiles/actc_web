import re
import pandas as pd
import json

import torch
from transformers import AutoTokenizer, AutoModel

def clean_text(text, mode=0):
    """
    清洗文本，适用于网站输入的古汉语文本
    :param text: 文本
    :param mode: 清洗模式
        - 0: 去掉所有标点符号和特殊字符，保留分句标点
        - 1: 保留基本句读标点（。！？），去掉其他标点符号
        - 2: 温和清洗，保留句读和部分标点，主要去掉格式字符
        - 3: 仅去掉多余空白和格式字符，保留所有标点
    :return: 清洗后的文本
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 先统一处理换行符和多余空白
    text = re.sub(r'\r\n|\r|\n', '', text)  # 去掉所有换行符
    text = re.sub(r'\s+', '', text)  # 去掉所有空白字符
    
    if mode == 0:
        # 去掉所有标点符号，但保留基本分句标点
        pattern = re.compile(r'[◎○▲#，；：、（）《》{}\"\'【】〖〗「」『』〔〕［］()""''‹›«»‚„‛‟¡¿§¶†‡•‰′″‴‵‶‷‸‹›※‼⁇⁈⁉⁏⁐⁑⁒⁓⁔⁕⁖⁗⁘⁙⁚⁛⁜⁝⁞⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔ\t\r ]')
    elif mode == 1:
        # 保留句读标点（。！？），去掉其他标点符号
        pattern = re.compile(r'[◎○▲#，；：、（）《》{}\"\'【】〖〗「」『』〔〕［］()""''‹›«»‚„‛‟¡¿§¶†‡•‰′″‴‵‶‷‸‹›※‼⁇⁈⁉⁏⁐⁑⁒⁓⁔⁕⁖⁗⁘⁙⁚⁛⁜⁝⁞⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔ\t\r ]')
    elif mode == 2:
        # 温和清洗，保留句读和逗号，去掉括号、引号等
        pattern = re.compile(r'[◎○▲#（）《》{}\"\'【】〖〗「」『』〔〕［］()""''‹›«»‚„‛‟¡¿§¶†‡•‰′″‴‵‶‷‸‹›※‼⁇⁈⁉⁏⁐⁑⁒⁓⁔⁕⁖⁗⁘⁙⁚⁛⁜⁝⁞⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔ\t\r ]')
    elif mode == 3:
        # 仅去掉特殊符号和格式字符，保留所有中文标点
        pattern = re.compile(r'[◎○▲#{}\"\'【】〖〗「」『』〔〕［］\t\r ]')
    else:
        # 默认使用mode 0
        pattern = re.compile(r'[◎○▲#，；：、（）《》{}\"\'【】〖〗「」『』〔〕［］()""''‹›«»‚„‛‟¡¿§¶†‡•‰′″‴‵‶‷‸‹›※‼⁇⁈⁉⁏⁐⁑⁒⁓⁔⁕⁖⁗⁘⁙⁚⁛⁜⁝⁞⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔ\t\r ]')
    
    cleaned_text = re.sub(pattern, '', text)
    
    # 最后再次清理可能残留的多余空白
    cleaned_text = re.sub(r'\s+', '', cleaned_text)
    
    return cleaned_text

# 其余函数保持不变...
def extract_fre_dict(table:dict, start, end):
     """
     提取词频字典：从已经有的大频率表中抽取出一部分排名的表，返回一个字典
     :param table: 词频表
     :param start: 起始排名
     :param end: 结束排名
     :return: 提取后的词频表
     """
     sort_table = sorted(table.items(), key=lambda x: x[1], reverse=True)
     return {k: v for k, v in sort_table[start:end]}

def safe_divide(numerator, denominator):
    """
    安全除法
    :param numerator: 分子
    :param denominator: 分母
    :return: 分子/分母
    """
    if denominator == 0 or denominator == 0.0:
        index = 0
    else: index = numerator/denominator
    return index


if __name__ == '__main__':
    # file_path = './data/3label_fewshot_baseline_result.jsonl'
    # df = parse_jsonl(file_path)
    #  print(df.head())
    sample_text = '古之学者必有师，师者，所以传道授业解惑者也。'
    print(clean_text(sample_text, 2))

        
    