# 古代汉语文本分类与复杂度分析系统

## 项目简介

这是一个基于机器学习的古代汉语文本分类与复杂度分析系统，能够对古代汉语文本进行多层次的复杂度评估，包括字词常用度、词汇多样性、句式复杂度和语体典雅度四个维度的分析。

## 主要功能

- **文本复杂度分析**：从词汇、句式、语体等多个维度评估古代汉语文本的复杂度
- **四层次评分**：提供字词常用度、词汇多样性、句式复杂度、语体典雅度的详细评分
- **可视化展示**：生成直观的图表展示分析结果
- **异常文本检测**：识别可能存在问题的文本内容
- **Web界面**：提供友好的网页界面进行文本分析

## 项目结构

```
Ancient Chinese Text Classification web/
├── app.py                      # Flask Web应用主文件
├── text_feature_extractor.py   # 文本特征提取器
├── text_level_analyzer.py      # 文本层次分析器
├── text_anomaly_detector.py    # 文本异常检测器
├── train_lightGBM.py          # LightGBM模型训练脚本
├── util.py                    # 工具函数
├── constant_variable.py       # 常量定义
├── pyproject.toml             # 现代Python项目配置文件
├── data/                      # 数据文件夹
│   ├── features_v3.xlsx       # 特征数据
│   └── train_vec.npz          # 训练向量
├── models/                    # 模型文件夹
│   ├── classifier_lgb.pkl     # LightGBM分类器
│   ├── tfidf_model.pkl       # TF-IDF模型
│   ├── modern_GPT2/          # 现代汉语GPT2模型
│   └── tradition_GPT2/       # 古代汉语GPT2模型
├── static/                    # 静态资源
│   ├── css/
│   │   └── style.css         # 样式文件
│   └── js/
│       └── main.js           # JavaScript文件
├── templates/                 # HTML模板
│   └── index.html            # 主页模板
└── tables/                   # 数据表文件
    ├── 汉语大词典_义项表.tsv
    ├── AncientChineseBigramFrequency.json
    └── AncientChineseCharacterFrequency.json
```

## 核心模块说明

### 1. `app.py` - Web应用主文件
- 基于Flask框架的Web应用
- 提供文本分析的HTTP接口
- 集成所有分析模块

### 2. `text_feature_extractor.py` - 特征提取器
- 提取21个文本特征维度
- 包括字数、词频、句长、语言模型困惑度等
- 支持古代汉语和现代汉语的对比分析

### 3. `text_level_analyzer.py` - 层次分析器
- 将特征按四个层次分类分析
- 基于分位数进行0-10分评分
- 生成可视化图表和详细报告

### 4. `text_anomaly_detector.py` - 异常检测器
- 检测文本中的异常内容，现已包括
- 识别可能的数据质量问题

### 5. `train_lightGBM.py` - 模型训练
- 使用LightGBM算法训练分类模型
- 支持特征选择和模型优化

## 环境搭建

### 方法一：使用 uv

`uv` 是一个极快的Python包管理器，比pip快10-100倍，是现代Python项目的最佳选择。

#### 优势：
- ⚡ **极快速度**：比pip快10-100倍
- 🔒 **锁定文件**：确保依赖版本一致性
- 🛡️ **可靠性**：更好的依赖解析
- 🚀 **现代化**：支持最新的Python打包标准

#### 安装步骤：

1. **安装 uv**
   ```bash
   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Linux/Mac
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # 或使用pip安装
   pip install uv
   ```

2. 放置GPT2模型

    古汉语GPT2:https://huggingface.co/uer/gpt2-chinese-ancient

    现代汉语GPT2:https://huggingface.co/uer/gpt2-chinese-cluecorpussmall

    下载后将模型文件分别放在`models/tradition_GPT2/`和`models/modern_GPT2/`目录下。

3. **一键运行**
```
cd 当前项目根目录
uv run app.py
```

## 特征说明

### 字词常用度（word_level）
- 总字数、字符平均频率、平均字义数
- 高频词比例、常用实词/虚词比例
- 古今常用字差异

### 词汇多样性（word_variety）
- 单次出现字比例
- 字符多样性指标（MATTR）

### 句式复杂度（sentence_level）
- 平均句长
- 句长变异度
- 短句长变异度

### 语体典雅度（discourse_level）
- 古今困惑度差异
- 体现文本的古典性和时代特征

## 数据文件说明

- `汉语大词典_义项表.tsv`：汉语大词典义项表
- `AncientChineseCharacterFrequency.json`：基于《四库全书》语料的古代汉语字频统计
- `AncientChineseBigramFrequency.json`：基于《四库全书》语料的古代汉语二元组频率
- `features_v3.xlsx`：特征数据集，用于计算分位数

## 模型文件

- `classifier_lgb.pkl`：训练好的LightGBM分类器
- `tfidf_model.pkl`：TF-IDF向量化模型
- `modern_GPT2/`：现代汉语GPT2语言模型
- `tradition_GPT2/`：古代汉语GPT2语言模型

## 项目打包分享

### 1. 创建环境配置文件

```bash
# 导出conda环境
conda env export > environment.yml

# 导出pip依赖
pip freeze > requirements_full.txt

# 使用uv导出依赖（推荐）
uv pip freeze > requirements_uv.txt
```

### 2. 打包整个项目

#### 推荐的项目分享结构：
```bash
# 创建项目压缩包，包含以下文件：
├── *.py                       # 所有Python源文件
├── requirements_minimal.txt   # 精简依赖
├── requirements_uv.txt        # uv专用依赖
├── pyproject.toml            # 现代Python项目配置
├── setup_uv.bat              # Windows一键安装脚本
├── setup_uv.sh               # Linux/Mac一键安装脚本
├── environment.yml           # conda环境配置（可选）
├── README.md                 # 项目说明文档
├── data/                     # 数据文件
├── models/                   # 模型文件
├── static/                   # 静态资源
├── templates/                # HTML模板
└── tables/                   # 数据表文件
```

#### 建议的分享方式：
1. **完整版**：包含所有文件和模型（适合直接运行）
2. **轻量版**：不包含models文件夹（需要用户自行训练或下载模型）
3. **源码版**：只包含源代码和配置文件

### 3. 接收者环境搭建

**方法A：使用conda环境文件**
```bash
conda env create -f environment.yml
conda activate ancient_chinese
```

**方法B：使用pip依赖文件**
```bash
conda create -n ancient_chinese python=3.9
conda activate ancient_chinese
pip install -r requirements_minimal.txt
```

**方法C：使用uv（最快速，推荐）**
```bash
# 安装uv（如果没有安装）
pip install uv

# 一键安装方式
# Windows
.\setup_uv.bat
# Linux/Mac
chmod +x setup_uv.sh && ./setup_uv.sh

# 或手动安装
uv venv .venv --python 3.9
# Windows
.venv\Scripts\activate
# Linux/Mac  
source .venv/bin/activate

# 快速安装依赖
uv pip install -r requirements_uv.txt
# 或从pyproject.toml安装
uv pip install -e .
```

**方法D：使用pyproject.toml（现代化方式）**
```bash
# 如果项目包含pyproject.toml文件
pip install -e .
# 或使用uv
uv pip install -e .
```

## 依赖版本说明

主要依赖包及其作用：

- **Flask 2.3.3**：Web框架
- **pandas 2.0.3**：数据处理
- **numpy 1.24.3**：数值计算
- **scikit-learn 1.3.0**：机器学习工具
- **matplotlib 3.7.2**：图表绘制
- **seaborn 0.12.2**：统计图表
- **lightgbm 4.0.0**：梯度提升算法
- **torch 2.0.1**：深度学习框架
- **transformers 4.33.2**：预训练模型

## 文件大小说明

- **models文件夹**：约 1-2GB（包含GPT2模型）
- **完整项目**：约 1.5-2.5GB
- **不含模型的项目**：约 10-50MB

💡 **分享建议**：
- 如果网络条件有限，建议分享不含models文件夹的版本
- 接收者可以通过训练脚本重新生成模型
- 或提供模型文件的单独下载链接

## 常见问题

### 1. 模型文件缺失
确保`models/`目录下的所有模型文件都已下载并放置正确位置。

### 2. 中文字体显示问题
系统需要支持中文字体，如SimHei、Microsoft YaHei等。

### 3. 内存不足
深度学习模型可能需要较大内存，建议至少8GB RAM。

### 4. 依赖包冲突
建议使用虚拟环境避免包冲突问题。

## 技术特色

1. **多维度分析**：从词汇到语篇的全方位文本复杂度评估
2. **古汉语专业性**：基于古代汉语语料库的专业分析
3. **可视化友好**：提供直观的图表和报告
4. **模块化设计**：各功能模块独立，便于扩展
5. **Web化部署**：支持在线使用和API调用

## 更新日志

- v1.0：初始版本，支持基础的四层次文本分析
- 后续版本将添加更多特征和改进分析算法

## 联系方式

如有问题或建议，请联系项目维护者。

## 许可证

[在此添加许可证信息]
