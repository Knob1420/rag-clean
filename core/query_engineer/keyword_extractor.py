"""
中文查询关键词提取 — 组合疑问结构清除 + 分词 + 去噪 + 提取关键实体词 + 词权重分类

融合 ragflow 的 rmWWW（组合疑问结构正则清除）和 rag-clean 的停用词过滤：
1. rmWWW 正则：处理"是什么样的""怎么办""是什么"等上下文组合疑问结构
2. 停用词集合：兜底零散虚词/疑问词残留
3. jieba 分词 + 领域术语注入
4. 全角→半角 + 中英文边界加空格
5. TermWeighter 3 档权重分类（HIGH/MEDIUM/LOW）
"""

import re
from pathlib import Path

import jieba

# ── 组合疑问结构正则（源自 ragflow QueryBase.rmWWW）──────────────

_RM_WWW_PATTERNS = [
    # 中文组合疑问结构：是*(怎么办|什么样的|...)是* → 整体删除
    # 例如："是什么样的" → ""、"怎么办" → ""、"是多少" → ""
    (
        r"是*(怎么办|什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*",
        "",
    ),
    # 英文疑问词 + be 动词缩写
    (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
    # 英文停用词/虚词
    (
        r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ",
        " ",
    ),
]


# ── 停用词（兜底 rmWWW 残留的零散虚词）────────────────────────────

_STOP_WORDS = frozenset({
    "您", "你", "我", "他", "她", "它", "这", "那", "我们",
    "的", "了", "在", "是", "就", "有", "于", "及", "即", "为", "最", "从", "以", "将", "与", "吧", "中", "又", "还",
    "把", "被", "让", "给", "对", "向", "往", "到", "得", "地",
    "和", "或", "但", "而", "且", "还是",
    "个", "只", "件", "条", "种", "些", "很", "也", "都", "颗", "家", "款", "次",
    "可以", "能", "会", "要", "需", "需要", "应该", "写", "写一下", "来", "去", "采用", "发射",
    "号", "大于", "小于", "等于",
    "相关", "关于", "比较", "介绍", "说明", "描述", "信息", "简单", "推荐", "分别", "适合", "具体", "主要",
    "问题", "解决", "即可", "一段话", "一下", "一个", "一段", "一段时间", "一款", "哪种", "或者", "哪几种",
    "几家", "几次", "几种", "超过", "翻译成", "以内", "之前",
    "颗卫星",
    "支持", "使用", "包括", "提供", "实现", "进行", "采用", "具备", "满足", "达到", "包含",
})


# ── 文本归一化 ─────────────────────────────────────────────────


def _fullwidth_to_halfwidth(text: str) -> str:
    """全角字符转半角（ASCII 范围）。"""
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:  # 全角空格
            result.append(" ")
        else:
            result.append(ch)
    return "".join(result)


def _add_boundary_spaces(text: str, protected_terms: set[str] | None = None) -> str:
    """在中英文/数字边界处加空格，方便后续分词。

    protected_terms: 已注册的领域术语（如 "X射线偏振探测器"），
                     加空格时跳过这些术语的内部边界。
    """
    if protected_terms:
        # 先把领域术语用占位符保护起来（大小写不敏感匹配）
        placeholders = {}
        text_lower = text.lower()
        for i, term in enumerate(sorted(protected_terms, key=len, reverse=True)):
            ph = f" __PROT{i}__ "
            placeholders[ph] = term
            idx = text_lower.find(term.lower())
            while idx >= 0:
                text = text[:idx] + ph + text[idx + len(term) :]
                text_lower = text.lower()
                idx = text_lower.find(term.lower(), idx + len(ph))

    # (ENG+NUM) + ZH  →  "NX1智算机" → "NX1 智算机"
    text = re.sub(r"([A-Za-z]+[0-9]+)([\u4e00-\u9fa5]+)", r"\1 \2", text)
    # ENG + ZH  →  "智加G" → "智加 G"
    text = re.sub(r"([A-Za-z])([\u4e00-\u9fa5]+)", r"\1 \2", text)
    # ZH + (ENG+NUM)  →  "智算机NX1" → "智算机 NX1"
    text = re.sub(r"([\u4e00-\u9fa5]+)([A-Za-z]+[0-9]+)", r"\1 \2", text)
    # ZH + ENG  →  "功耗W" → "功耗 W"
    text = re.sub(r"([\u4e00-\u9fa5]+)([A-Za-z])", r"\1 \2", text)

    if protected_terms:
        # 还原占位符
        for ph, term in placeholders.items():
            text = text.replace(ph, term)

    return text


def _rm_www(txt: str) -> str:
    """清除组合疑问结构（rmWWW）。

    与逐词停用词过滤互补：
    - rmWWW 能处理 "是什么样的" "怎么办" "是什么" 等上下文组合
    - 停用词过滤只能逐词匹配，无法处理这类组合
    """
    otxt = txt
    for pattern, replacement in _RM_WWW_PATTERNS:
        txt = re.sub(pattern, replacement, txt, flags=re.IGNORECASE)
    if not txt.strip():
        txt = otxt  # 全删光则回退原文
    return txt


# ── 短英文型号名/系列名保护 ──────────────────────────────────────────

# 型号模式: G1, G2, NX1, NX2, MN300, 3D 等
_MODEL_PATTERN = re.compile(r"^[A-Z]{1,3}\d{1,4}[A-Z]?$", re.IGNORECASE)
# 系列前缀模式: NX, G, X 等独立出现的系列名
_SERIES_PREFIX_PATTERN = re.compile(r"^[A-Z]{1,3}$", re.IGNORECASE)


def _is_model_name(token: str) -> bool:
    """判断是否为短型号名（如 G1, NX1, 3D），应保留不被丢弃。"""
    return bool(_MODEL_PATTERN.match(token))


# ── 主类 ──────────────────────────────────────────────────────


class ChineseKeywordExtractor:
    """融合版中文查询关键词提取

    处理流程:
    1. 中英文边界加空格
    2. 全角→半角 + 小写
    3. 组合疑问结构清除（rmWWW）
    4. 去除标点
    5. jieba 分词
    6. 停用词过滤 + 单字/纯数字过滤
    """

    def __init__(self):
        # 加载领域术语到 jieba（确保"智算机"等不被拆散）
        self._domain_terms: set[str] = set()
        self._load_domain_terms()
        jieba.initialize()

    def _load_domain_terms(self):
        """从 terms_seed.json 加载领域术语到 jieba 分词器。

        加载位置：data/terms_seed.json
        """
        import json

        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / "data" / "terms_seed.json"
        with open(path, encoding="utf-8") as f:
            terms = json.load(f)
        for term in terms:
            jieba.add_word(term, freq=1000)
            self._domain_terms.add(term)
            # 同时注册小写版本（查询归一化后是小写）
            term_lower = term.lower()
            if term_lower != term:
                jieba.add_word(term_lower, freq=1000)
                self._domain_terms.add(term_lower)

    def extract(self, query: str) -> list[tuple[str, float]]:
        """
        从查询中提取关键词并附带权重。

        输入: "请问NX1智算机的重量是多少"
        输出: [("nx1", 3.0), ("智算机", 3.0), ("重量", 1.5)]
        """
        # 1. 中英文边界加空格（保护领域术语不被拆散）
        text = _add_boundary_spaces(query, self._domain_terms)

        # 2. 全角→半角 + 小写
        text = _fullwidth_to_halfwidth(text).lower()

        # 3. 组合疑问结构清除（rmWWW）
        text = _rm_www(text)

        # 4. 去除标点（保留字母数字和中文字符）
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return []

        # 5. jieba 分词
        tokens = list(jieba.cut(text))

        # 6. 过滤：去停用词 + 去单字（型号名/系列前缀除外）+ 去纯数字短于2位
        keywords: list[str] = []
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            if t in _STOP_WORDS:
                continue
            # 型号名保留（如 g1, nx1, 3d）
            if _is_model_name(t):
                keywords.append(t)
                continue
            # 系列前缀保留（如 nx, g, x — 来自 "NX系列" "G系列"）
            if _SERIES_PREFIX_PATTERN.match(t):
                keywords.append(t)
                continue
            # 单字中文通常无意义
            if len(t) == 1 and re.match(r"[\u4e00-\u9fa5]", t):
                continue
            # 单字英文字母（非系列前缀）跳过
            if len(t) == 1 and re.match(r"[a-z]$", t):
                continue
            # 1-2 位纯数字跳过（除非是型号的一部分）
            if re.match(r"^\d{1,2}$", t):
                continue
            keywords.append(t)

        # 7. 词权重分类
        from core.query_engineer.term_weight import get_term_weighter
        weighted = get_term_weighter().classify_keywords(keywords)
        return weighted


# ── 全局实例 ──────────────────────────────────────────────────

_extractor: ChineseKeywordExtractor | None = None


def get_keyword_extractor() -> ChineseKeywordExtractor:
    """获取关键词提取器单例"""
    global _extractor
    if _extractor is None:
        _extractor = ChineseKeywordExtractor()
    return _extractor
