# SciFact Neural Retrieval Lab

面向科学事实核验的可训练检索系统：从 BM25 基线出发，训练领域双塔检索器，使用混合召回、Cross-Encoder 重排和句子级证据选择，最后通过自适应路由决定回答、重试或 NEI。

项目重点不是把摘要塞进向量库，而是把 **hard negative、训练目标、候选排序、证据定位和服务延迟** 放进同一套可复现实验链路。

## 结果概览

官方 SciFact corpus 包含 5,183 篇摘要；训练集 809 条 claim，开发集 300 条 claim。以下结果均来自同一官方 dev split：

| 方法 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 78.74% | 64.16% | 67.16% |
| Zero-shot MiniLM Dense | 77.43% | 57.84% | 62.39% |
| Fine-tuned Dense（hard negative） | 80.32% | 63.86% | 67.54% |
| BM25 + Dense fixed RRF | **84.06%** | **68.97%** | **71.95%** |
| RRF + domain Cross-Encoder | 84.06% | **67.03%** | **69.40%** |

领域 Cross-Encoder 在 BM25 Top-10 候选上重排，MRR@10 从 64.16% 提升到 67.03%，nDCG@10 从 67.16% 提升到 69.40%。重排不改变候选集合，因此 Recall@10 保持不变。

服务侧对 5,183 篇摘要建立向量索引：

| Backend | Recall@10 | P95 | Index size |
|---|---:|---:|---:|
| Qdrant local | 81.40% | 12.74 ms | 20.46 MiB |
| Milvus | 81.40% | 12.16 ms | 38.14 MiB |

固定 100 请求、预热 10、并发 8 的端到端 benchmark：`/v1/retrieve` 为 59.17 QPS、P95 162.80 ms；`/v1/rag` 为 2.61 QPS、P95 3.63 s。

## 系统架构

```text
SciFact corpus / claims
        |
        +--> BM25 ------------------------------+
        |                                        |
        +--> Dense Retriever --------------------+--> weighted RRF
                    ^                             |
                    |                             +--> Cross-Encoder rerank
          hard-negative InfoNCE                   |
                                                  +--> sentence evidence selector
                                                          |
                                                          +--> answer / retry / NEI
```

在线链路额外提供 query planning 和 adaptive routing：

```text
claim -> deterministic rewrite/decomposition -> multi-query retrieval
      -> top-1 / margin / coverage / support-conflict features
      -> answer | retry | abstain
```

## 核心模块

| 模块 | 作用 |
|---|---|
| `retrieval_lab/bm25.py` | 倒排索引、BM25 打分和批量检索 |
| `retrieval_lab/neural.py` | Transformer 双塔、mean pooling、InfoNCE 和 dense retrieval |
| `retrieval_lab/negatives.py` | BM25 hard-negative mining、gold 排除和 teacher margin 过滤 |
| `retrieval_lab/rerank.py` | Cross-Encoder pair scoring 与候选重排 |
| `retrieval_lab/query_planning.py` | 可审计 query rewrite/decomposition 与 provenance |
| `retrieval_lab/adaptive_routing.py` | 基于证据覆盖和冲突特征的 answer/retry/abstain 路由 |
| `retrieval_lab/vector_store.py` | Qdrant 与 Milvus collection 后端 |
| `retrieval_lab/knowledge_router.py` | 可选 Wikipedia 背景知识通道，保留 source provenance |
| `retrieval_lab/serving.py` | Embedding、混合召回、重排、证据抽取和 FastAPI |
| `retrieval_lab/metrics.py` | Recall、MRR、nDCG、证据句命中和 run 文件 |

## 训练与检索

双塔训练使用 SciFact train split：正样本为标注支持 claim 的文献，难负样本来自 BM25 高分但未标注支持的文献；同一 query 的负例经过 gold 排除，训练时启用 in-batch negatives。一次可复现运行挖掘出 6,472 个 BM25 候选，排除 703 次正例命中，并为 809 条训练 claim 保留 8 个 hard negatives。

训练后的 dense encoder 与 BM25 通过 fixed weighted RRF 融合。Cross-Encoder 使用 919 个正例和 2,427 个 BM25 难负例完成 1 epoch 领域微调，重排阶段只处理已召回候选。

## 证据定位

抽取式证据链路在官方句级标注上进行独立评测：

| 指标 | 结果 |
|---|---:|
| 300 条 claim 的引用文献 Top-10 命中率 | 85.67% |
| 188 条带句级标注 claim 的标注文献 Top-10 命中率 | 94.68% |
| Top-3 证据句至少命中一条金标句 | 76.06% |
| 366 条金标证据句覆盖率 | 53.28% |

```bash
PYTHONPATH=. python scripts/evaluate_evidence_grounding.py \
  --data-root data/scifact \
  --run results/quality_ablation/hybrid_rrf_hn.run.jsonl \
  --retrieval-k 10 --snippets-per-document 3 \
  --output results/quality_ablation/evidence_grounding_hybrid_rrf_hn.json
```

## 前沿组件消融

公开 checkpoint 的 ColBERT v2 和 SPLADE v2 在相同 BM25 Top-20 候选上做真实重排：

| 方法 | Recall@10 | MRR@10 | nDCG@10 | CPU ms/query |
|---|---:|---:|---:|---:|
| BM25 candidate order | 78.74% | 64.16% | 67.16% | - |
| ColBERT v2 rerank | 74.31% | 60.17% | 62.97% | 1,542 |
| SPLADE v2 rerank | 76.59% | 63.19% | 65.70% | 1,936 |

两种通用模型在该候选池上没有超过 BM25，结果保留用于模型选择和延迟权衡。CRAG-style retry 使用领域 Cross-Encoder、dense retry 和 DeBERTa sentence NLI，在固定 calibration/evaluation 划分上触发 15.87% 的重试；evaluation 的 Recall@10/MRR@10/nDCG@10 为 78.08%/66.77%/68.62%，低于默认混合链路，因此当前不作为默认路由。

## 服务 API

启动服务后提供：

| Endpoint | 作用 |
|---|---|
| `GET /healthz` | 服务状态 |
| `POST /v1/embed` | 文本向量化 |
| `POST /v1/plan` | 返回 rewrite、subquery 和 provenance |
| `POST /v1/route` | 返回 answer/retry/abstain 决策与特征 |
| `POST /v1/retrieve` | BM25、dense、RRF 检索 |
| `POST /v1/rerank` | Cross-Encoder 重排 |
| `POST /v1/rag` | 返回结构化证据抽取结果 |

## 快速开始

环境：Python 3.10+。SciFact 官方数据目录：

```text
data/scifact/
  corpus.jsonl
  claims_train.jsonl
  claims_dev.jsonl
  claims_test.jsonl
```

安装并运行 BM25：

```bash
python -m pip install -e ".[dev]"
python -m retrieval_lab.cli bm25 --config configs/scifact_bm25.json
```

挖掘 hard negatives、训练双塔并评测：

```bash
python -m retrieval_lab.cli mine-negatives \
  --config configs/scifact_mine_negatives.json
python -m retrieval_lab.cli train-dual \
  --config configs/scifact_dual_encoder.json
python -m retrieval_lab.cli dense-eval \
  --data-root data/scifact \
  --claims-file claims_dev.jsonl \
  --checkpoint checkpoints/scifact-minilm-infonce \
  --output-metrics results/dense_scifact_dev.json
```

构建 Qdrant local index 并启动 API：

```bash
python scripts/build_vector_index.py \
  --data-root data/scifact \
  --qdrant-path results/qdrant_local \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
python scripts/serve_rag.py \
  --data-root data/scifact \
  --qdrant-path results/qdrant_local
```

使用 Qdrant 服务模式：

```bash
docker compose up -d qdrant
python scripts/build_vector_index.py \
  --data-root data/scifact \
  --qdrant-url http://127.0.0.1:6333
docker compose up -d rag-api
```

Milvus 通过 `retrieval_lab.vector_store.MilvusStore` 接入，安装
`pymilvus>=2.4` 后传入 Milvus URI、collection 名称和向量维度即可复用同一套
`upsert/search` 接口；Milvus 与 Qdrant 的对比结果见上方 benchmark 表。

运行服务压测：

```bash
python scripts/benchmark_service.py \
  --data-root data/scifact --endpoint /v1/rag \
  --requests 100 --concurrency 8 \
  --output results/benchmark_rag_c8.json
```

## 实验协议

- 训练使用 train，模型选择、路由阈值和重试阈值使用 dev；test 只在最终运行时读取。
- 每次运行保存配置、随机种子、训练 loss、完整 run、指标 JSON 和阶段耗时。
- 文献级指标与句子级证据指标分开报告；检索召回、重排质量、引用支持率和服务延迟不混为单一分数。
- Wikipedia 仅作为带来源标识的背景通道，不写入 SciFact gold evidence 或训练标签。

## 测试

```bash
python scripts/audit_scifact.py \
  --data-root data/scifact \
  --output results/dataset_audit.json
python -m unittest discover -s tests -v
```

## 数据与许可证

SciFact claims/evidence annotations 使用 CC BY 4.0，abstracts 来自 S2ORC（ODC-By 1.0）。本仓库不提交原始数据和模型权重；代码使用 MIT License。
