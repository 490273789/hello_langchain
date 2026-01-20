from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

# 定义切割规则 用 ""兜底
separators = [
    r"\n#{1,6}\s",  # 标题
    r"\n\d+(?:\.\d+)*\s",  # 数字编号标题 1. / 2.3. 等
    "\n\n",  # 段落
    "\n",  # 行
    " ",  # 空格
    "",  # 兜底字符级
]
splitter = RecursiveCharacterTextSplitter(
    separators=separators,
    chunk_size=70,
    chunk_overlap=10,
    is_separator_regex=True,  # 告诉分割器上面包含正则
)

text = """# 1.3 基于递归字符分块
## 分块策略：给定一组由“粗到细”的分隔符（如段落→换行→空格→字符），自上而下递归切分，在不超出 chunk_size 的前提下尽量保留自然语义边界。
## 优点：在“保持语义边界”和“控制块大小”之间取得稳健平衡，对大多数文本即插即用。
## 缺点：分隔符配置不当会导致块粒度失衡，极度格式化文本（表格/代码）效果一般。
## 适用场景：综合性语料、说明文档、报告、知识库条目。"""

chunks = splitter.split_text(text)
print(chunks)
