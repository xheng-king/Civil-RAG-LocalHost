# settings.py

# pre-training embedding
# base_url_set = ""
# embedding_model = ""
# embedding_API_key = ""

# fine_tunned embedding
base_url_set = ""
embedding_model = ""
embedding_API_key = ""

rerank_model = ""
rerank_base_url = ""
rerank_API_key = ""

vllm = ""
vllm_base_url = ""
vllm_API_key = ""

llm = ""
llm_base_url = ""
llm_API_key = ""

# 基础检索参数 (当 ENABLE_ADAPTIVE_RETRIEVAL = False 时使用，或作为第一轮的基础值)
BASE_INITIAL_RETRIEVE_K = 5   # ChromaDB 初次召回数量
BASE_FINAL_TOP_K = 3          # 重排序后保留给 LLM 的数量

# 是否启用自适应检索重试机制 (类似 FLARE/Iterative Retrieval)
# True: 如果回答被判定为不正确，则增加召回数量并重新查询，最多重试 MAX_RETRIEVAL_ROUNDS 次
# False: 仅执行单次标准查询流程，使用上方的 BASE_ 参数
ENABLE_ADAPTIVE_RETRIEVAL = False 

#最大重试轮次
MAX_RETRIEVAL_ROUNDS = 2

# 每轮增加的召回数量
RETRIEVAL_STEP_SIZE = 5

# 重排序后额外保留给LLM的数量增量 (即每轮多给LLM看1个文档)
RERANK_OUTPUT_STEP_SIZE = 1