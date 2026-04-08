# 查询集格式说明

## 查询集 JSON 格式

```json
{
  "metadata": {
    "name": "查询集名称",
    "version": "1.0"
  },
  "queries": [
    {
      "id": "q001",
      "query": "用户问题",
      "question_type": "事实"
    }
  ]
}
```

## 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `id` | 是 | 查询唯一标识，格式建议 `q001`, `q002` ... |
| `query` | 是 | 用户原始问题 |
| `question_type` | 是 | 问题类型，见下方枚举 |

## 问题类型枚举

| 类型 | 说明 | 示例 |
|------|------|------|
| `事实` | 事实性查询，通常有明确答案 | "G1多重？" |
| `单项` | 针对单一对象的功能/操作查询 | "如何部署智算机集群？" |
| `对比` | 涉及两个或多个对象的对比 | "G3和NX3有什么区别？" |
| `模糊` | 意图模糊或开放性查询 | "三体计算星座是什么？" |
| `推荐` | 需求导向的推荐查询 | "有没有适合边缘计算的产品？" |
| `其他复杂` | 多条件、多步骤的复杂查询 | "全系产品功耗对比和散热差异？" |

## 标注流程

1. 运行 `python -m eval.eval_pipeline export --queries eval/datasets/queries.json --output eval/exports/run_001.json`
2. 打开导出的 JSON 文件
3. 在每条 query 的 `relevant_chunk_ids` 数组中填入正确的 chunk_id
4. 将文件另存为 `run_001_labeled.json`
5. 运行 `python -m eval.eval_pipeline evaluate --labeled eval/exports/run_001_labeled.json`
