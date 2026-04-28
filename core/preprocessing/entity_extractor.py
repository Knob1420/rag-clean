"""
步骤2 — 全域实体 / 别名 / 术语自动抽取

能力：
1. LLM Prompt 抽取通用实体（PRODUCT/PROJECT/ORG/TECH/PERSON/LOC/TIME）
2. 规则白词 + 正则抽取领域实体（PRODUCT/PROJECT/TECH）
3. 共现聚类（Embedding 相似度 + 共现频率）自动发现别名候选

输出：
- entity_raw.json: 原始实体库
- alias_candidates.json: 别名候选（待人工确认）
"""

import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np

# ── LLM 实体抽取 ─────────────────────────────────────────────


def _get_llm_client():
    """获取 LLM 客户端（延迟初始化）"""
    try:
        from core.generation.llm import get_llm_client

        return get_llm_client()
    except ImportError:
        return None


def _extract_llm_entities(text: str) -> list[dict]:
    """
    用 LLM Prompt 抽取通用实体（PRODUCT/PROJECT/ORG/TECH/PERSON/LOC/TIME）。

    相比 HanLP NER，LLM 对中文领域专有名词（产品型号、项目名）识别效果更好。
    """
    from prompt import ENTITY_EXTRACTION_SYSTEM_PROMPT, build_entity_extraction_prompt
    from core.generation.llm import parse_json_response

    llm_client = _get_llm_client()
    if llm_client is None:
        return []

    prompt = build_entity_extraction_prompt(text)
    messages = [
        {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        response = llm_client.call(messages, temperature=0.3)
        result = parse_json_response(response)
        entities = result.get("entities", [])
        if not isinstance(entities, list):
            return []
        # 过滤无效实体
        return [
            e
            for e in entities
            if isinstance(e, dict) and e.get("text") and e.get("entity_type")
        ]
    except Exception as e:
        from loguru import logger

        logger.warning(f"LLM 实体抽取失败: {e}")
        return []


async def _extract_llm_entities_batch(
    texts: list[str], max_concurrency: int = 10
) -> list[list[dict]]:
    """
    批量 LLM 实体抽取（asyncio 并发）。
    """
    from prompt import ENTITY_EXTRACTION_SYSTEM_PROMPT, build_entity_extraction_prompt
    from core.generation.llm import parse_json_response

    llm_client = _get_llm_client()
    if llm_client is None:
        return [[] for _ in texts]

    async def _run() -> list[list[dict]]:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _call_one(idx: int, text: str) -> tuple[int, list[dict]]:
            async with semaphore:
                prompt = build_entity_extraction_prompt(text)
                messages = [
                    {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                try:
                    response = await llm_client._call_async(messages)
                    result = parse_json_response(response)
                    entities = result.get("entities", [])
                    if not isinstance(entities, list):
                        return idx, []
                    return idx, [
                        e
                        for e in entities
                        if isinstance(e, dict)
                        and e.get("text")
                        and e.get("entity_type")
                    ]
                except Exception as e:
                    from loguru import logger

                    logger.warning(f"LLM 批量实体抽取失败 [{idx}]: {e}")
                    return idx, []

        tasks = [_call_one(i, t) for i, t in enumerate(texts)]
        results = await asyncio.gather(*tasks)
        return [r for _, r in sorted(results, key=lambda x: x[0])]

    return await _run()


# ── 领域白词配置 ─────────────────────────────────────────────


def _load_domain_terms() -> tuple[set[str], dict[str, str]]:
    """
    加载领域白词（从 terms_seed.json）。
    返回 (标准名集合, 别名→标准名映射)
    """
    terms_path = Path(__file__).parent / "scripts" / "terms_seed.json"
    if not terms_path.exists():
        return set(), {}

    with open(terms_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    standard_names = set(data.values())  # 标准名集合
    alias_to_standard = {}
    for alias, standard in data.items():
        if alias != standard:  # 别名才记录
            alias_to_standard[alias] = standard

    return standard_names, alias_to_standard


_STANDARD_NAMES, _ALIAS_TO_STANDARD = _load_domain_terms()

# ── Embedding 用于别名发现 ──────────────────────────────────


def _get_embedder():
    """获取 embedding 工具（复用自己的 embedder）"""
    try:
        from core.client.embedder import encode_batch

        return encode_batch
    except ImportError:
        return None


# ── 实体类型常量 ────────────────────────────────────────────


class EntityType:
    PRODUCT = "PRODUCT"  # 产品名、型号
    PROJECT = "PROJECT"  # 项目名
    ORG = "ORG"  # 机构/企业名
    TECH = "TECH"  # 技术术语
    PERSON = "PERSON"  # 人名
    LOC = "LOC"  # 地名
    TIME = "TIME"  # 时间


# ── 规则白词抽取 ─────────────────────────────────────────────


def _extract_domain_entities_by_rules(text: str) -> list[dict]:
    """
    用规则白词抽取领域实体（PRODUCT/PROJECT/TECH）。

    方法：
    1. 基于 terms_seed.json 白词精确匹配
    2. 基于正则的型号抽取（G1/G2/NX1/NX2 等）
    3. 基于正则的项目名抽取（三体计算星座 等）
    """
    entities = []

    # 1. 白词匹配（PRODUCT 为主）
    # 注意：不用 \b 词边界，因为在"智加G1"这种中英混合文本里边界不可靠
    # 遍历所有别名（alias→standard）和所有标准名（standard→standard）两种都匹配
    all_terms = dict(_ALIAS_TO_STANDARD)  # alias → standard
    for std in _STANDARD_NAMES:
        if std not in all_terms:  # 标准名自身也要匹配
            all_terms[std] = std

    for term, standard in all_terms.items():
        # 子串匹配（简单高效）
        lower_text = text.lower()
        lower_term = term.lower()
        start = 0
        while True:
            pos = lower_text.find(lower_term, start)
            if pos == -1:
                break
            # 判断类型
            entity_type = _infer_domain_entity_type(standard)
            entities.append(
                {
                    "text": standard,  # 归一到标准名
                    "entity_type": entity_type,
                    "start": pos,
                    "end": pos + len(term),
                }
            )
            start = pos + 1  # 继续找下一个（允许重叠匹配）

    # 2. 型号正则（如 G1/G2/G3/NX1/NX2/NX3/NX4）
    model_pattern = r"\b([A-Z][0-9]{1,2}(?:\s*[/-]\s*[A-Z][0-9]{1,2})*)\b"
    for m in re.finditer(model_pattern, text):
        model_text = m.group(1).strip()
        if model_text and not _is_common_abbreviation(model_text):
            entities.append(
                {
                    "text": model_text,
                    "entity_type": EntityType.PRODUCT,
                    "start": m.start(),
                    "end": m.end(),
                }
            )

    # 3. 项目名正则（如 三体计算星座、智能遥感计算卫星）
    project_patterns = [
        r"三体计算星座",
        r"智能遥感计算卫星[一二三号零零]+",
        r"太空计算组件",
        r"星座[一二三]+号",
        r"MN300[-\s]?[0-9]+",
    ]
    for pattern in project_patterns:
        for m in re.finditer(pattern, text):
            entities.append(
                {
                    "text": m.group(0),
                    "entity_type": EntityType.PROJECT,
                    "start": m.start(),
                    "end": m.end(),
                }
            )

    return entities


def _infer_domain_entity_type(standard_name: str) -> str:
    """根据标准名推断实体类型"""
    if any(
        kw in standard_name
        for kw in [
            "星载智能计算机",
            "星载激光通信机",
            "星载路由器",
            "智桥",
            "智光",
            "智加G",
            "智加NX",
            "智加X",
            "G1",
            "G2",
            "G3",
            "NX1",
            "NX2",
            "NX3",
            "NX4",
            "X1",
            "X2",
            "X3",
            "X4",
        ]
    ):
        return EntityType.PRODUCT
    if any(kw in standard_name for kw in ["星座", "卫星", "MN"]):
        return EntityType.PROJECT
    if any(kw in standard_name for kw in ["计算", "通信", "处理", "系统"]):
        return EntityType.TECH
    return EntityType.TECH


def _is_common_abbreviation(text: str) -> bool:
    """判断是否通用缩写（非领域专有）"""
    common = {"API", "CPU", "GPU", "DNA", "RNA", "CEO", "CTO", "COO", "PDF", "MD"}
    return text.upper() in common


# ── HanLP NER 抽取（已废弃，保留作兼容性） ──────────────────


# ── 实体合并与去重 ───────────────────────────────────────────


def _merge_entities(all_entities: list[dict]) -> dict[str, dict]:
    """
    合并多个来源的实体，按 (text, entity_type) 分组去重。
    返回 {标准名: entity_record}
    """
    entity_map = {}

    for ent in all_entities:
        key = (ent["text"], ent["entity_type"])
        if key in entity_map:
            entity_map[key]["frequency"] += ent.get("frequency", 1)
            old_chunks = set(entity_map[key].get("source_chunks", []))
            new_chunks = set(ent.get("source_chunks", []))
            entity_map[key]["source_chunks"] = list(old_chunks | new_chunks)
        else:
            entity_map[key] = {
                "entity_name": ent["text"],
                "entity_type": ent["entity_type"],
                "frequency": ent.get("frequency", 1),
                "source_docs": [],
                "source_chunks": list(ent.get("source_chunks", [])),
            }

    return entity_map


# ── 别名发现（Embedding 语义相似度）──────────────────────────


EMBEDDING_SIM_THRESHOLD = 0.85


def _discover_alias_by_embedding(
    entity_list: list[str],
    embedder,
) -> list[dict]:
    """
    用 Embedding 语义相似度发现别名候选。

    逻辑：
    1. 计算每对实体的 embedding
    2. 相似度 > EMBEDDING_SIM_THRESHOLD → 自动绑定候选别名
    3. 由调用方确认后生效
    """
    if not entity_list or embedder is None:
        return []

    # 计算 embedding
    embeddings = embedder(entity_list)
    if not embeddings:
        return []

    # 转 numpy
    vecs = np.array(embeddings)
    n = len(entity_list)
    candidates = []

    for i in range(n):
        for j in range(i + 1, n):
            # cosine similarity
            v1, v2 = vecs[i], vecs[j]
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                continue
            sim = float(np.dot(v1, v2) / (norm1 * norm2))

            if sim >= EMBEDDING_SIM_THRESHOLD:
                # 判断谁是主名、谁是别名（频率高的为主名）
                candidates.append(
                    {
                        "standard_name": entity_list[i],
                        "alias": entity_list[j],
                        "embedding_sim": round(sim, 3),
                    }
                )

    return candidates


def _group_alias_candidates(
    embedding_candidates: list[dict],
) -> dict[str, list[dict]]:
    """
    按 standard_name 分组别名候选。

    返回 {standard_name: [{alias, embedding_sim}, ...]}
    """
    result: dict[str, list[dict]] = defaultdict(list)
    for cand in embedding_candidates:
        result[cand["standard_name"]].append(
            {
                "alias": cand["alias"],
                "embedding_sim": cand["embedding_sim"],
            }
        )
    return result


# ── 实体抽取主流程 ──────────────────────────────────────────


async def extract_entities(
    chunks: list[dict],
    ner_mode: Literal["llm", "hanlp", "none"] = "llm",
) -> tuple[list[dict], list[dict]]:
    """
    全域实体抽取主流程（异步版本）。

    Args:
        chunks: chunk 列表，每个 chunk 是 dict：
            {
                "content": str,      # chunk 文本
                "chunk_id": str,      # chunk ID
                "doc_id": str,        # 文档 ID
                "doc_title": str,     # 文档标题
                "source_file": str,    # 来源文件
            }
        ner_mode: NER 模式选择
            - "llm"（默认）：使用 LLM Prompt 抽取，对领域专有名词效果更好
            - "hanlp"：使用 HanLP NER（已废弃，效果较差）
            - "none"：仅使用规则白词抽取

    Returns:
        (entity_raw_list, alias_candidates_list)
        - entity_raw_list: 原始实体库
        - alias_candidates_list: 别名候选列表
    """
    from loguru import logger

    logger.info(
        f"[EntityExtractor] 开始抽取实体，chunks={len(chunks)}, ner_mode={ner_mode}"
    )

    all_entities: list[dict] = []

    # 1. 规则白词抽取（领域实体，每个 chunk 独立处理）
    for chunk in chunks:
        text = chunk["content"]
        chunk_id = chunk["chunk_id"]
        domain_entities = _extract_domain_entities_by_rules(text)
        for ent in domain_entities:
            ent["source_chunks"] = [chunk_id]
            all_entities.append(ent)

    # 2. NER 抽取（通用实体）
    if ner_mode == "llm":
        # 批量 LLM 抽取，并发 10
        texts = [c["content"] for c in chunks]
        llm_results = await _extract_llm_entities_batch(texts, max_concurrency=10)
        for chunk, entities in zip(chunks, llm_results):
            chunk_id = chunk["chunk_id"]
            for ent in entities:
                ent["source_chunks"] = [chunk_id]
                all_entities.append(ent)
        logger.info(f"[EntityExtractor] LLM 批量抽取完成，chunks={len(chunks)}")
    elif ner_mode == "hanlp":
        logger.warning("HanLP NER 已废弃，效果较差，建议使用 ner_mode='llm'")

    # 3. 合并去重
    entity_map = _merge_entities(all_entities)

    # 4. 构建 entity_raw_list
    entity_raw_list = list(entity_map.values())

    # 5. 别名发现（Embedding 语义相似度）
    embedder = _get_embedder()
    entity_names = [e["entity_name"] for e in entity_raw_list]

    embedding_candidates = _discover_alias_by_embedding(entity_names, embedder)
    alias_grouped = _group_alias_candidates(embedding_candidates)

    # 构建 alias_candidates_list
    alias_candidates_list = []
    for standard_name, candidates in alias_grouped.items():
        alias_candidates_list.append(
            {
                "standard_name": standard_name,
                "candidate_aliases": candidates,
            }
        )

    logger.info(
        f"[EntityExtractor] 完成: {len(entity_raw_list)} 实体, "
        f"{len(alias_candidates_list)} 个别名候选"
    )

    return entity_raw_list, alias_candidates_list


# ── 保存结果 ────────────────────────────────────────────────


def save_results(
    entity_raw_list: list[dict],
    alias_candidates_list: list[dict],
    output_dir: str,
    merge: bool = False,
) -> dict[str, str]:
    """
    保存抽取结果到 JSON 文件。

    Args:
        merge: 为 True 时，先加载已有文件并合并，再保存（支持增量批次处理）
    """
    from loguru import logger

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    entity_raw_path = output_path / "step2_entity_raw.json"
    alias_candidates_path = output_path / "step2_alias_candidates.json"

    if merge and entity_raw_path.exists() and alias_candidates_path.exists():
        with open(entity_raw_path, "r", encoding="utf-8") as f:
            old_entity_raw = json.load(f)
        with open(alias_candidates_path, "r", encoding="utf-8") as f:
            old_alias_candidates = json.load(f)

        entity_raw_list = _merge_entity_raw(entity_raw_list, old_entity_raw)
        alias_candidates_list = _merge_alias_candidates(
            alias_candidates_list, old_alias_candidates
        )
        logger.info(
            f"[EntityExtractor] 增量合并后: {len(entity_raw_list)} 实体, "
            f"{len(alias_candidates_list)} 别名候选"
        )

    with open(entity_raw_path, "w", encoding="utf-8") as f:
        json.dump(entity_raw_list, f, ensure_ascii=False, indent=2)

    with open(alias_candidates_path, "w", encoding="utf-8") as f:
        json.dump(alias_candidates_list, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[EntityExtractor] 结果已保存: {entity_raw_path}, {alias_candidates_path}"
    )

    return {
        "entity_raw": str(entity_raw_path),
        "alias_candidates": str(alias_candidates_path),
    }


def _merge_entity_raw(
    new_list: list[dict],
    old_list: list[dict],
) -> list[dict]:
    """合并新旧 entity_raw，按 (entity_name, entity_type) 去重。"""
    merged: dict[tuple, dict] = {}

    for ent in old_list:
        key = (ent["entity_name"], ent["entity_type"])
        merged[key] = dict(ent)

    for ent in new_list:
        key = (ent["entity_name"], ent["entity_type"])
        if key in merged:
            merged[key]["frequency"] += ent.get("frequency", 1)
            # 合并 source_chunks（去重）
            old_chunks = set(merged[key].get("source_chunks", []))
            new_chunks = set(ent.get("source_chunks", []))
            merged[key]["source_chunks"] = list(old_chunks | new_chunks)
        else:
            merged[key] = dict(ent)

    return list(merged.values())


def _merge_alias_candidates(
    new_list: list[dict],
    old_list: list[dict],
) -> list[dict]:
    """合并新旧 alias_candidates，按 standard_name 分组，alias 去重保留更高 embedding_sim。"""
    grouped: dict[str, dict] = {}

    for item in old_list:
        sn = item["standard_name"]
        grouped[sn] = {"standard_name": sn, "candidate_aliases": []}
        for alias in item.get("candidate_aliases", []):
            grouped[sn]["candidate_aliases"].append(alias)

    for item in new_list:
        sn = item["standard_name"]
        if sn not in grouped:
            grouped[sn] = {"standard_name": sn, "candidate_aliases": []}

        existing_aliases = {a["alias"] for a in grouped[sn]["candidate_aliases"]}
        for alias_entry in item.get("candidate_aliases", []):
            alias_name = alias_entry["alias"]
            sim = alias_entry.get("embedding_sim", 0)
            if alias_name not in existing_aliases:
                grouped[sn]["candidate_aliases"].append(alias_entry)
                existing_aliases.add(alias_name)
            else:
                # 相同 alias 已存在，保留更高 sim
                for existing in grouped[sn]["candidate_aliases"]:
                    if existing["alias"] == alias_name and sim > existing.get(
                        "embedding_sim", 0
                    ):
                        existing["embedding_sim"] = sim

    return list(grouped.values())
