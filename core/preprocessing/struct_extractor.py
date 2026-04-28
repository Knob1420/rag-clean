"""
步骤3 — 结构化专项抽取（product_params + cooperation）

重构后只保留 2 类抽取：
1. product_params：嫁接 products_specs.json 标准库，从 entity_raw 按产品名过滤相关 chunk，
                   LLM 只补充空字段（功能/在轨/场景）
2. cooperation：从 entity_raw 的 ORG 实体出发，找含 2+ 单位的 chunk，LLM 抽关系，增量去重

输出：
- step3_product_params.json
- step3_cooperation.json
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger


# ── LLM 客户端 ──────────────────────────────────────────────


def _get_llm_client():
    """获取 LLM 客户端（延迟导入避免循环依赖）"""
    try:
        from core.generation.llm import get_llm_client

        return get_llm_client()
    except ImportError:
        return None


# ── 加载 products_specs.json 标准库 ────────────────────────


def _load_products_specs(path: Path) -> dict:
    """加载产品参数标准库。"""
    if not path.exists():
        logger.warning(f"[StructExtractor] products_specs.json 不存在: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 构建 entity_raw 映射 ───────────────────────────────────


def _build_product_chunk_map(entity_raw: list[dict]) -> dict[str, list[str]]:
    """
    从 entity_raw 构建 entity_name → source_chunks 映射（只取 PRODUCT 类型）。

    Returns:
        {"智加G1": ["chunk_id_1", "chunk_id_2", ...], ...}
    """
    product_map: dict[str, list[str]] = {}
    for ent in entity_raw:
        if ent.get("entity_type") == "PRODUCT":
            name = ent["entity_name"]
            chunks = ent.get("source_chunks", [])
            if name not in product_map:
                product_map[name] = []
            product_map[name].extend(chunks)
    return product_map


def _build_org_chunk_map(entity_raw: list[dict]) -> dict[str, list[str]]:
    """
    从 entity_raw 构建 entity_name → source_chunks 映射（只取 ORG 类型）。

    Returns:
        {"国星宇航": ["chunk_id_1", ...], ...}
    """
    org_map: dict[str, list[str]] = {}
    for ent in entity_raw:
        if ent.get("entity_type") == "ORG":
            name = ent["entity_name"]
            chunks = ent.get("source_chunks", [])
            if name not in org_map:
                org_map[name] = []
            org_map[name].extend(chunks)
    return org_map


# ── 查找 cooperation 候选 chunk ────────────────────────────


def _find_cooperation_candidates(
    org_map: dict[str, list[str]],
    chunks: list[dict],
    primary_org: str = "之江实验室",
) -> list[tuple[set[str], str, str]]:
    """
    找同时含 primary_org + 其他 ORG 实体的 chunk。

    Args:
        primary_org: 主体合作方（默认之江实验室）

    Returns:
        [(org_set, chunk_id, chunk_content), ...]
    """
    # 构建 chunk_id → chunk_content 映射
    chunk_content_map = {c["chunk_id"]: c["content"] for c in chunks}

    # 构建 org_name → 出现在哪些 chunk 中
    org_name_to_chunk_ids: dict[str, set[str]] = {}
    for org_name, chunk_ids in org_map.items():
        for cid in chunk_ids:
            if cid not in org_name_to_chunk_ids:
                org_name_to_chunk_ids[cid] = set()
            org_name_to_chunk_ids[cid].add(org_name)

    candidates = []
    for chunk_id, org_names in org_name_to_chunk_ids.items():
        # 必须同时包含 primary_org 和至少一个其他 ORG
        if primary_org not in org_names:
            continue
        other_orgs = org_names - {primary_org}
        if len(other_orgs) >= 1:
            content = chunk_content_map.get(chunk_id, "")
            if content:
                candidates.append((org_names, chunk_id, content))

    return candidates


# ── 构建 product_params 抽取任务 ───────────────────────────


def _build_model_tasks(
    products_specs: dict, chunk_map: dict[str, list[str]], chunks: list[dict]
) -> list[dict]:
    """
    为 products_specs.json 中每个型号构建 LLM 抽取任务。

    只为"功能"/"在轨情况"/"适用场景"这些可能为空的动态字段生成 LLM prompt。
    已有值的字段不覆盖。

    Returns:
        [{"model": "智加G1", "category": "...", "series": "...",
          "existing_params": {...}, "relevant_chunks": [...],
          "empty_fields": [...], "source_chunk_ids": [...]}, ...]
    """
    tasks = []
    chunk_content_map = {c["chunk_id"]: c["content"] for c in chunks}

    # 需要 LLM 补充的动态字段
    dynamic_fields = {"功能", "在轨情况", "适用场景", "场景"}

    for category, series_list in products_specs.items():
        if not isinstance(series_list, list):
            continue
        for series_entry in series_list:
            series_name = series_entry.get("series", "")
            model_list = series_entry.get("model_list", [])
            if not isinstance(model_list, list):
                continue
            for model_entry in model_list:
                model_name = model_entry.get("model", "")
                if not model_name:
                    continue

                params = model_entry.get("params", {})
                if not isinstance(params, dict):
                    params = {}

                # 找该型号在 entity_raw 中关联的 source_chunks
                relevant_chunk_ids = chunk_map.get(model_name, [])
                relevant_contents = []
                for cid in relevant_chunk_ids:
                    content = chunk_content_map.get(cid, "")
                    if content:
                        relevant_contents.append(content)

                # 确定哪些动态字段为空
                empty_fields = [f for f in dynamic_fields if not params.get(f)]

                task = {
                    "model": model_name,
                    "category": category,
                    "series": series_name,
                    "existing_params": dict(params),
                    "relevant_chunks": relevant_contents,
                    "empty_fields": empty_fields,
                    "source_chunk_ids": relevant_chunk_ids,
                }
                tasks.append(task)

    return tasks


# ── LLM Prompt ─────────────────────────────────────────────


_PRODUCT_PARAMS_FILL_PROMPT = """你是一个太空计算业务专家，负责补充产品参数中为空的关键字段。

## 你的任务
从提供的上下文文本中，为指定型号提取以下字段的值：
- 功能：在轨功能描述（如 L0-L4数据处理、在轨推理等）
- 在轨情况：在轨验证记录（如发射时间、卫星名称、在轨状态）
- 适用场景：典型应用场景

## 重要规则
1. **只填充空字段**：已有的参数值（合作单位、架构、算力等）**不要修改**
2. **只提取文本中明确提到的信息**，不要推测或编造
3. 如果某字段在提供的上下文中没有提到，该字段保持空字符串
4. 优先使用包含该型号具体参数描述的 chunk 内容

## 输入信息
型号：{model_name}
已有参数：
{existing_params}

待填充字段：{empty_fields}

相关文档上下文（来自 entity_raw 中该型号关联的 source_chunks）：
{chunk_contexts}

## 输出格式
输出一个 JSON 对象，包含以下字段：
{{
  "params": {{
    "功能": "如果上下文中提到了功能则填写，否则为空字符串",
    "在轨情况": "如果上下文中提到了在轨验证情况则填写，否则为空字符串",
    "适用场景": "如果上下文中提到了适用场景则填写，否则为空字符串"
  }},
  "filled_fields": ["实际填充了哪些字段"],
  "confidence": 0.0-1.0置信度
}}

只输出 JSON，不要有其他文字。
"""


_COOPERATION_EXTRACT_PROMPT = """你是一个太空计算业务专家，负责从文档片段中提取「之江实验室」与其他单位的合作关系。

## 输入
文档片段中出现了以下单位：{org_names}

文档内容：
{chunk_content}

## 你的任务
从上述文档片段中提取与「之江实验室」的合作关系，输出 JSON：

{{
  "units": ["之江实验室", "合作单位2", ...],  // 之江实验室固定在第一位，第二个单位来自上述列表
  "content": "合作内容摘要（简短描述合作的内容）",
  "products_or_projects": ["涉及的产品或项目名称", ...],  // 从文档中识别与该合作相关的产品或项目
  "source_chunk": "来源chunk_id",
  "confidence": 0.0-1.0置信度
}}

## 规则
1. units 必须来自输入中提供的单位列表，**不要自己创造单位名**
2. **units 第一个固定为「之江实验室」**，合作单位在后面
3. 只提取与「之江实验室」有**真实合作关系**的信息，背景介绍/新闻引用/政府公告中出现的单位**不算**
4. 判断标准：文档中明确提到双方签署协议、合作研发、合作建设等实质性合作行为
5. 如果只是背景介绍（如"之江实验室与XX单位曾在某会议共同参加"）不算合作，units 返回 ["之江实验室"]
6. 置信度根据信息完整度评估（0.5=背景提及，0.7=部分合作，0.9=完整合作）
7. 输出只包含 JSON，不要有其他文字
"""


# ── 解析 LLM JSON 响应 ─────────────────────────────────────


def _parse_json_response(response: str) -> dict | None:
    """解析 LLM 返回的 JSON 响应。"""
    import re

    # 尝试直接解析
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 尝试从 ```json 代码块中提取
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取花括号内容
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ── product_params LLM 填充 ────────────────────────────────


async def _fill_product_params_batch(
    tasks: list[dict], llm_client, max_concurrency: int = 10
) -> list[dict]:
    """
    并发为多个型号填充空参数。

    Returns:
        填充结果列表，每个元素为 {"model": ..., "params": {...}, "filled_fields": [...], "confidence": ...}
    """
    if not tasks or llm_client is None:
        return []

    async def _call_one(task: dict) -> dict:
        model_name = task["model"]
        existing_params = task["existing_params"]
        empty_fields = task["empty_fields"]
        chunk_contexts = task["relevant_chunks"]

        if not empty_fields or not chunk_contexts:
            return {
                "model": model_name,
                "params": {},
                "filled_fields": [],
                "confidence": 0.0,
            }

        # 拼接上下文（限制总长度）
        context_text = "\n\n---\n\n".join(chunk_contexts[:5])  # 最多 5 个 chunk
        if len(context_text) > 3000:
            context_text = context_text[:3000] + "..."

        # 构建已有参数的展示文本
        params_text = "\n".join(
            f"  {k}: {v}" for k, v in existing_params.items()
        )

        prompt = _PRODUCT_PARAMS_FILL_PROMPT.format(
            model_name=model_name,
            existing_params=params_text,
            empty_fields=", ".join(empty_fields),
            chunk_contexts=context_text,
        )

        try:
            response = await llm_client._call_async(
                [
                    {"role": "system", "content": "你是一个太空计算业务专家。"},
                    {"role": "user", "content": prompt},
                ],
            )
            result = _parse_json_response(response)
            if result and "params" in result:
                return {
                    "model": model_name,
                    "params": result["params"],
                    "filled_fields": result.get("filled_fields", []),
                    "confidence": result.get("confidence", 0.5),
                }
        except Exception as e:
            logger.warning(f"[StructExtractor] product_params LLM 失败 [{model_name}]: {e}")

        return {
            "model": model_name,
            "params": {},
            "filled_fields": [],
            "confidence": 0.0,
        }

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_one(task: dict):
        async with semaphore:
            return await _call_one(task)

    results = await asyncio.gather(*[_run_one(t) for t in tasks])
    return list(results)


# ── cooperation LLM 抽取 ───────────────────────────────────


async def _extract_cooperation_batch(
    candidates: list[tuple[set[str], str, str]],
    llm_client,
    max_concurrency: int = 10,
) -> list[dict]:
    """
    并发为多个候选 chunk 抽取合作关系。

    Args:
        candidates: [(org_set, chunk_id, chunk_content), ...]
        llm_client: LLM 客户端

    Returns:
        抽取结果列表
    """
    if not candidates or llm_client is None:
        return []

    async def _call_one(org_set: set[str], chunk_id: str, chunk_content: str) -> dict:
        org_names = sorted(org_set)
        # 限制 chunk 长度
        content = chunk_content[:3000] if len(chunk_content) > 3000 else chunk_content

        prompt = _COOPERATION_EXTRACT_PROMPT.format(
            org_names=", ".join(org_names),
            chunk_content=content,
        )

        try:
            response = await llm_client._call_async(
                [
                    {"role": "system", "content": "你是一个太空计算业务专家。"},
                    {"role": "user", "content": prompt},
                ],
            )
            result = _parse_json_response(response)
            if result and isinstance(result, dict):
                result["source_chunk"] = chunk_id
                return result
        except Exception as e:
            logger.warning(f"[StructExtractor] cooperation LLM 失败 [{chunk_id}]: {e}")

        return {
            "units": [],
            "content": "",
            "products_or_projects": [],
            "source_chunk": chunk_id,
            "confidence": 0.0,
        }

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_one(item):
        org_set, chunk_id, chunk_content = item
        async with semaphore:
            return await _call_one(org_set, chunk_id, chunk_content)

    results = await asyncio.gather(*[_run_one(c) for c in candidates])
    return list(results)


# ── 主流程 ─────────────────────────────────────────────────


async def extract_product_params(
    products_specs: dict,
    entity_raw: list[dict],
    chunks: list[dict],
    llm_enabled: bool = True,
    max_concurrency: int = 10,
) -> list[dict]:
    """
    product_params 抽取主流程。

    逻辑：
    1. 从 products_specs.json 加载标准库
    2. 从 entity_raw 构建 product → source_chunks 映射
    3. 为每个型号构建任务（已有参数不覆盖，只补充空字段）
    4. LLM 并发填充

    Returns:
        同 products_specs.json 层级结构，params 被补充
    """
    logger.info(
        f"[StructExtractor] extract_product_params 开始，models={sum(len(s.get('model_list', [])) for si in products_specs.values() if isinstance(si, list) for s in si)}"
    )

    # 构建 product → source_chunks 映射
    product_chunk_map = _build_product_chunk_map(entity_raw)

    # 构建型号任务
    tasks = _build_model_tasks(products_specs, product_chunk_map, chunks)
    logger.info(f"[StructExtractor] 构建了 {len(tasks)} 个型号任务")

    if not tasks:
        return []

    # LLM 填充
    llm_client = _get_llm_client() if llm_enabled else None
    if llm_client is None:
        logger.warning("[StructExtractor] LLM 不可用，跳过 product_params 填充")
        # 返回原始结构，不做任何填充
        return _build_result_from_specs(products_specs, [])

    filled_results = await _fill_product_params_batch(
        tasks, llm_client, max_concurrency
    )

    # 合并到标准库结构
    result = _merge_filled_params(products_specs, filled_results)

    logger.info(f"[StructExtractor] product_params 完成，填充了 {len([r for r in filled_results if r.get('filled_fields')])} 个型号")

    return result


async def extract_cooperation(
    entity_raw: list[dict],
    chunks: list[dict],
    llm_enabled: bool = True,
    max_concurrency: int = 10,
) -> list[dict]:
    """
    cooperation 抽取主流程。

    逻辑：
    1. 从 entity_raw 提取 ORG 实体
    2. 找同时含 2+ ORG 的 chunk
    3. LLM 并发抽取合作关系
    4. 按 frozenset(units) 去重

    Returns:
        [{"units":[], "content":"", "products_or_projects":[], "source_chunk":"", "confidence":0.0}, ...]
    """
    logger.info(f"[StructExtractor] extract_cooperation 开始，entity_raw={len(entity_raw)}")

    # 构建 ORG → source_chunks 映射
    org_map = _build_org_chunk_map(entity_raw)
    logger.info(f"[StructExtractor] ORG 实体数量: {len(org_map)}")

    # 找含 2+ ORG 的 chunk
    candidates = _find_cooperation_candidates(org_map, chunks)
    logger.info(f"[StructExtractor] cooperation 候选 chunk 数: {len(candidates)}")

    if not candidates:
        return []

    # LLM 抽取
    llm_client = _get_llm_client() if llm_enabled else None
    if llm_client is None:
        logger.warning("[StructExtractor] LLM 不可用，跳过 cooperation 抽取")
        return []

    results = await _extract_cooperation_batch(candidates, llm_client, max_concurrency)

    # 去重（按 frozenset(sorted(units))）
    deduped = _dedupe_cooperation(results)

    logger.info(f"[StructExtractor] cooperation 完成，去重后: {len(deduped)} 条")

    return deduped


# ── 辅助函数 ───────────────────────────────────────────────


def _build_result_from_specs(products_specs: dict, _filled_results: list[dict]) -> list[dict]:
    """用空填充结果构建输出结构（当 LLM 不可用时）。"""
    result = []
    for category, series_list in products_specs.items():
        if not isinstance(series_list, list):
            continue
        for series_entry in series_list:
            series_name = series_entry.get("series", "")
            model_list = series_entry.get("model_list", [])
            if not isinstance(model_list, list):
                continue
            for model_entry in model_list:
                result.append({
                    "category": category,
                    "series": series_name,
                    "model": model_entry.get("model", ""),
                    "params": model_entry.get("params", {}),
                    "confidence": 0.0,
                })
    return result


def _merge_filled_params(products_specs: dict, filled_results: list[dict]) -> list[dict]:
    """
    将 LLM 填充结果合并回 products_specs 层级结构。

    已有值不覆盖，只填充空字段。
    """
    # 构建 model → filled_params 映射
    filled_map = {r["model"]: r["params"] for r in filled_results if r.get("params")}

    result = []
    for category, series_list in products_specs.items():
        if not isinstance(series_list, list):
            continue
        for series_entry in series_list:
            series_name = series_entry.get("series", "")
            model_list = series_entry.get("model_list", [])
            if not isinstance(model_list, list):
                continue
            for model_entry in model_list:
                model_name = model_entry.get("model", "")
                params = dict(model_entry.get("params", {}))

                # 合并填充结果（已有值不覆盖）
                if model_name in filled_map:
                    for k, v in filled_map[model_name].items():
                        if v and not params.get(k):  # 只填充空值
                            params[k] = v

                result.append({
                    "category": category,
                    "series": series_name,
                    "model": model_name,
                    "params": params,
                    "confidence": next(
                        (r["confidence"] for r in filled_results if r["model"] == model_name),
                        0.0,
                    ),
                })

    return result


def _dedupe_cooperation(results: list[dict]) -> list[dict]:
    """
    按 frozenset(sorted(units)) 去重合作记录。

    新的组合追加，已有的合并 content 和 source_chunks。
    """
    seen: set[frozenset] = set()
    deduped_map: dict[frozenset, dict] = {}

    for r in results:
        units = r.get("units", [])
        if not units:
            continue

        key = frozenset(sorted(units))
        if key not in seen:
            seen.add(key)
            deduped_map[key] = dict(r)
            deduped_map[key]["source_chunks"] = [r.get("source_chunk", "")]
        else:
            # 合并：content 拼接，source_chunks 追加
            existing = deduped_map[key]
            content = r.get("content", "")
            if content and content != existing.get("content", ""):
                existing["content"] = existing.get("content", "") + "；" + content
            if r.get("source_chunk"):
                existing["source_chunks"].append(r["source_chunk"])
            # 合并 products_or_projects
            existing_projs = set(existing.get("products_or_projects", []))
            for p in r.get("products_or_projects", []):
                if p:
                    existing_projs.add(p)
            existing["products_or_projects"] = sorted(list(existing_projs))
            # 取最高 confidence
            if r.get("confidence", 0) > existing.get("confidence", 0):
                existing["confidence"] = r["confidence"]

    return list(deduped_map.values())


# ── 保存结果 ───────────────────────────────────────────────


def save_results(
    product_params: list[dict],
    cooperation: list[dict],
    output_dir: str,
    merge: bool = False,
) -> dict[str, str]:
    """
    保存抽取结果到 JSON 文件。

    Args:
        merge: 为 True 时，先加载已有文件并增量合并，再保存

    Returns:
        {"product_params": path, "cooperation": path}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    product_params_path = output_path / "step3_product_params.json"
    cooperation_path = output_path / "step3_cooperation.json"

    if merge:
        if product_params_path.exists():
            with open(product_params_path, "r", encoding="utf-8") as f:
                old_product_params = json.load(f)
            product_params = _merge_product_params(old_product_params, product_params)

        if cooperation_path.exists():
            with open(cooperation_path, "r", encoding="utf-8") as f:
                old_cooperation = json.load(f)
            cooperation = _merge_cooperation(old_cooperation, cooperation)

    with open(product_params_path, "w", encoding="utf-8") as f:
        json.dump(product_params, f, ensure_ascii=False, indent=2)

    with open(cooperation_path, "w", encoding="utf-8") as f:
        json.dump(cooperation, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[StructExtractor] 结果已保存: {product_params_path}, {cooperation_path}"
    )

    return {
        "product_params": str(product_params_path),
        "cooperation": str(cooperation_path),
    }


def _merge_product_params(old: list[dict], new: list[dict]) -> list[dict]:
    """
    合并新旧 product_params。

    按 category→series→model 路径合并；基础库字段被 LLM 填充覆盖（基础库空值被覆盖，已有值不覆盖）。
    """
    # 按 (category, series, model) 建立索引
    merged: dict[tuple, dict] = {}

    for item in old:
        key = (item.get("category", ""), item.get("series", ""), item.get("model", ""))
        merged[key] = dict(item)

    for item in new:
        key = (item.get("category", ""), item.get("series", ""), item.get("model", ""))
        if key in merged:
            # 合并 params（已有值不覆盖）
            old_params = merged[key].get("params", {})
            new_params = item.get("params", {})
            merged_params = dict(old_params)
            for k, v in new_params.items():
                if v and not merged_params.get(k):
                    merged_params[k] = v
            merged[key]["params"] = merged_params

            # 置信度取更高
            if item.get("confidence", 0) > merged[key].get("confidence", 0):
                merged[key]["confidence"] = item["confidence"]
        else:
            merged[key] = dict(item)

    return list(merged.values())


def _merge_cooperation(old: list[dict], new: list[dict]) -> list[dict]:
    """
    合并新旧 cooperation 记录。

    按 frozenset(sorted(units)) 去重；新组合追加，已有的跳过。
    """
    seen: set[frozenset] = set()
    merged_list = []

    for item in old:
        units = item.get("units", [])
        if units:
            key = frozenset(sorted(units))
            seen.add(key)
            merged_list.append(item)

    for item in new:
        units = item.get("units", [])
        if not units:
            continue
        key = frozenset(sorted(units))
        if key not in seen:
            seen.add(key)
            merged_list.append(item)

    return merged_list
