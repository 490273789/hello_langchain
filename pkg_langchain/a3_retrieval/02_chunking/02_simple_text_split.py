from langchain_text_splitters import (
    CharacterTextSplitter,
)

# CharacterTextSplitter 简单切分
text = "这是第一段。\n\n这是第二段。\n这是第三段"
splitter = CharacterTextSplitter(separator="\n", chunk_size=10, chunk_overlap=2)

text = splitter.split_text(text)
print(text)
print("-" * 88)
print(len(text))

# 基于长度切分
splitter = CharacterTextSplitter(
    separator="",  # 纯按长度切
    chunk_size=60,  # 依据实验与模型上限调整
    chunk_overlap=9,  # 15% 重叠
)

text = """1.1 基于固定长度分块。
- 分块策略：按预设字符数 chunk_size 直接切分，不考虑文本结构。
- 优点：实现最简单、速度快、对任意文本通用。
- 缺点：容易破坏语义边界；块过大容易引入较多噪声，过小则会导致上下文不足。
- 适用场景：结构性弱的纯文本，或数据预处理初期的基线方案。"""
chunks = splitter.split_text(text)
