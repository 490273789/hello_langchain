import re


def split_sentences_zh(text: str):
    # 在句末标点（。！？；）后面带可选引号的场景断句
    pattern = re.compile(r"([^。！？；]*[。！？；]+|[^。！？；]+$)")
    sentences = [
        m.group(0).strip() for m in pattern.finditer(text) if m.group(0).strip()
    ]
    return sentences


def sentence_chunk(text: str, chunk_size=600, overlap=80):
    sents = split_sentences_zh(text)
    chunks, buf = [], ""
    for s in sents:
        if len(buf) + len(s) <= chunk_size:
            buf += s
        else:
            if buf:
                chunks.append(buf)
                # 简单重叠：从当前块尾部截取 overlap 字符与下一句拼接
            buf = (buf[-overlap:] if overlap > 0 and len(buf) > overlap else "") + s
    if buf:
        chunks.append(buf)
    return chunks


text = """1.2 基于句子的分块
- 分块策略：先按句子切分，再将若干句子聚合成满足chunk_size的块；保证最基本的语义完整性。
- 优点：句子级完整性最好。对问句/答句映射友好。便于高质量引用。
- 缺点：中文分句需特别处理。仅句子级切分可能导致块过短，需后续聚合。
- 适用场景：法律法规、新闻、公告、FAQ 等以句子为主的文本。
- 中文分句注意事项：
  - 不要直接用 NLTK 英文 Punkt：无法识别中文标点，分句会失败或异常。
  - 可以直接使用以下内容进行分句：
    - 基于中文标点的正则：按“。！？；”等切分，保留引号与省略号等边界。
    - 使用支持中文的 NLP 库进行更精细的分句：
    - HanLP（推荐，工业级，支持繁多语言学特性）Stanza（清华/斯坦福合作，中文支持较好）spaCy + pkuseg 插件（或 zh-core-web-sm/med/lg 生态）
    """


chunks = sentence_chunk(text, chunk_size=60, overlap=9)
for chunk in chunks:
    print(chunk, end="\n\n---\n")
