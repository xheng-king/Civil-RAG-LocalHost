# 土木工程规范智能检索增强生成 (RAG) 系统

## 1. 介绍

本项目是一个面向**土木工程领域**的智能问答系统，旨在解决传统规范文档结构复杂、专业术语密集及查询效率低的问题。系统基于 **检索增强生成 (RAG)** 架构，结合了自适应检索策略、重排序机制以及领域专用的嵌入模型微调技术。

### 核心特性
*   **领域专用知识库**：对于复杂文档如土木工程规范，可进行结构感知解析与智能分块处理。
*   **自适应检索机制 (Adaptive Retrieval)**：类似 FLARE框架，当初始检索生成的答案置信度较低时，系统自动扩大检索范围（增加召回文档数量），通过多轮迭代提升回答准确率。
*   **高精度重排序 (Rerank)**：引入交叉编码器重排序模型，对向量检索初步召回的结果进行语义相关性二次排序，显著提升上下文质量。
*   **多维评估体系**：支持 ACC (准确率), MRR (平均倒数排名), NDCG (归一化折损累计增益) 等指标的系统性评估。

---

## 2. 如何运行

### 环境准备

1.  **克隆项目**
    ```bash
    git clone <your-repo-url>
    cd rag_ce
    ```

2.  **创建虚拟环境并安装依赖**
    ```bash
    python3 -m venv rag_venv
    source rag_venv/bin/activate  # Linux/Mac
    # 或 rag_venv\Scripts\activate (Windows)
    
    pip install -r requirements.txt
    ```

3.  **配置 API 密钥与参数**
    编辑 `src/settings.py` 文件，填入必要的 API Key 和模型配置：
    ```python
    # src/settings.py
    
    # Embedding 模型配置 (例如使用 DashScope/Qwen)
    embedding_API_key = "your-embedding-api-key"
    base_url_set = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model = "text-embedding-v2" # 或您微调后的模型名称

    # LLM 配置 (例如使用 Qwen-Max/Turbo)
    llm_API_key = "your-llm-api-key"
    llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm = "qwen-max"

    # Rerank 模型配置
    rerank_API_key = "your-rerank-api-key"
    rerank_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    rerank_model = "gte-rerank" 

    # 功能开关
    ENABLE_ADAPTIVE_RETRIEVAL = False  # 设为 True 启用自适应多轮检索
    ```

### 启动交互界面

运行主程序进入命令行交互模式：

```bash
python src/main.py
```

在交互界面中，您可以：
1. 编辑数据库集合（查看、导出、清空、新建集合）
2. 文本嵌入索引（选择集合存储文本-向量对，非空则清空）
3. 问答查询（针对选定数据库集合进行问答）
4. Markdown文件图像处理（调用视觉模型为图片生成描述并附加）
5. 从文本文件生成问答对（用于测评系统）
6. 从问答对生成训练数据集（用于微调嵌入模型）

### 批量评估

若需对测试集进行批量性能评估：

```bash
python src/rag_interface.py
```
评估结果将保存至项目根目录下，请注意及时将评估结果移动至其他位置如/data/result，否则下次评估将对其覆盖