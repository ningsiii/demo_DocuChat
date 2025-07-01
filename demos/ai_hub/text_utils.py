import re
def full_clean(text: str) -> str:
    """清理 PDF 抽取常见噪声"""
    # 去掉全形空格、连字等
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    # 去掉私用区乱码（常见于 \ue***）
    text = re.sub(r"[\ue000-\uf8ff]", "", text)
    # 去掉连续空格
    text = re.sub(r"\s{2,}", " ", text)
    # 去掉页眉/页脚中“学习大语言模型原理必看的 10 篇论文 n”
    text = re.sub(r"学\s*习\s*大\s*语\s*言.*?论\s*文\s*\d+\s*", "", text)
    return text.strip()

def postprocess_docs(raw_docs, min_len: int = 80):
    """清洗 + 过滤 + 去重"""
    seen = set()
    cleaned = []
    for d in raw_docs:
        txt = full_clean(d["content"])
        # 去重
        sig = hash(txt)
        if len(txt) >= min_len and sig not in seen:
            cleaned.append({"content": txt, "source": d["source"]})
            seen.add(sig)
    return cleaned