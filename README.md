# SciFact Neural Retrieval Lab

一个面向可复现实验的科学文献检索项目。项目不是 RAG 页面或框架拼装，而是围绕检索模型的训练目标、负样本质量、假负例和两阶段排序开展对照实验。

## 当前真实状态

已完成并提供代码：

- 本地 SciFact `corpus.jsonl` 与 `claims_*.jsonl` 读取和字段校验。
- 从零实现的 BM25、倒排索引和批量检索。
- Recall@K、MRR@K、nDCG@K 统一评测与 run 文件格式。
- BM25 hard-negative mining，自动排除标注正例。
- Cross-Encoder 教师分数驱动的保守假负例过滤。
- 基于 Hugging Face Transformer 的共享/非共享参数双塔编码器。
- 自定义 query-positive-negative InfoNCE，支持 hard negatives、in-batch negatives 和 negative mask。
- 双塔训练 CLI、dense retrieval CLI，以及 Cross-Encoder rerank 接口和 CLI。
- FastAPI 服务化链路：MiniLM Embedding、Qdrant local-mode 向量索引、BM25 + Dense RRF 混合召回、Cross-Encoder Rerank 和返回 Top-3 证据句的抽取式 RAG 接口。
- 固定请求数和并发的 HTTP benchmark，记录 QPS、P50/P95/P99、成功率及分阶段耗时。
- 使用 SciFact 官方句级标注评测引用文献命中、证据文献命中和返回片段的证据句命中。

尚未完成或不能声称的结果：

- 已完成 MiniLM 双塔 hard-negative InfoNCE 训练和同一 Dev 集对照；该结果只代表本仓库固定配置，不声称优于 BGE、SciBERT 或论文结果。
- 已完成固定 RRF、query-adaptive RRF、通用 Cross-Encoder Top-50 和句子级 evidence-aware 诊断；其中后两者未带来稳定的 MRR 提升，作为负结果保留。
- 配置中的预训练模型只是可运行起点，不代表已经完成对应 GPU 实验。
- 当前服务指标是本地单机 benchmark，不是生产 SLO；生成端默认返回证据抽取结果，不声称 LLM 生成质量。

## 服务化运行与实测结果

使用官方 SciFact 语料构建 5,183 篇文档的 384 维 Embedding，并写入 Qdrant local-mode 持久化 collection：

```bash
python scripts/build_vector_index.py \
  --data-root data/scifact \
  --qdrant-path results/qdrant_local \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
python scripts/serve_rag.py \
  --data-root data/scifact \
  --qdrant-path results/qdrant_local
```

接口包括 `GET /healthz`、`POST /v1/embed`、`POST /v1/retrieve`、`POST /v1/rerank` 和 `POST /v1/rag`。Docker Compose 同时提供 Qdrant 服务模式配置；本次可复现实验使用 local mode，避免把未运行的多容器部署写成线上经验。

需要运行服务模式时，先启动 Qdrant，再通过 HTTP upsert 索引，最后启动 API：

```bash
docker compose up -d qdrant
python scripts/build_vector_index.py --data-root data/scifact \
  --qdrant-url http://127.0.0.1:6333
docker compose up -d rag-api
```

在官方 SciFact dev 的 300 条 claim 上，服务侧使用已构建的 zero-shot Qdrant 索引取得 Recall@10=84.10%、MRR@10=65.54%、nDCG@10=69.83%；离线训练后 Dense + BM25 RRF 的严格对照为 Recall@10=84.06%、MRR@10=68.97%、nDCG@10=71.95%。前者是服务快照，后者是质量优化实验，不能混写。

固定 100 请求、预热 10、并发 8 的最新本地压测如下：

| Endpoint | QPS | P50 | P95 | P99 | 成功率 |
|---|---:|---:|---:|---:|---:|
| `/v1/retrieve` | 59.17 | 131.58 ms | 162.80 ms | 175.78 ms | 100% |
| `/v1/rag` | 2.61 | 2.97 s | 3.63 s | 3.83 s | 100% |

端到端 RAG 的主要瓶颈是 CPU Cross-Encoder Rerank；后续可通过批量推理、轻量模型、GPU 部署和候选数控制降低 P95。完整结果保存在 `results/benchmark_retrieve_c8.json`、`results/benchmark_rag_c8.json` 和 `results/service_hybrid_scifact_dev.json`。

## 证据定位评测

固定 BM25 + 训练后双塔 RRF 的 run 文件在官方 SciFact dev 上另行进行证据评测。该评测仅衡量抽取式链路是否返回官方标注的引用文献及其摘要证据句，不将其等同于生成答案的正确性。

| 指标 | 结果 |
|---|---:|
| 300 条 claim 的引用文献 Top-10 命中率 | 85.67% |
| 188 条带句级证据标注 claim 的标注文献 Top-10 命中率 | 94.68% |
| 上述 claim 中，Top-3 证据句至少命中一条标注句 | 76.06% |
| 366 条金标证据句的覆盖率 | 53.28% |

结果保存于 `results/quality_ablation/evidence_grounding_hybrid_rrf_hn.json`，可按以下方式复跑：

```bash
PYTHONPATH=. python scripts/evaluate_evidence_grounding.py \
  --data-root data/scifact \
  --run results/quality_ablation/hybrid_rrf_hn.run.jsonl \
  --retrieval-k 10 --snippets-per-document 3 \
  --output results/quality_ablation/evidence_grounding_hybrid_rrf_hn.json
```

## 已复现的本地基线

在官方 SciFact dev split 上，使用 5,183 篇 corpus 文档和全部 300 条有标注 claim 实测。此次结果由工作区已有的官方 SciFact 数据副本计算，发布配置将同一数据集规范化为 `data/scifact`。BM25 参数为 `k1=1.5`、`b=0.75`、标题重复 2 次；分词为小写字母数字正则，无 stemming、无停用词表。

| K | Recall@K | MRR@K | nDCG@K |
|---:|---:|---:|---:|
| 1 | 0.5419 | 0.5600 | 0.5600 |
| 5 | 0.7276 | 0.6337 | 0.6503 |
| 10 | 0.7874 | 0.6416 | 0.6716 |
| 20 | 0.8292 | 0.6445 | 0.6829 |
| 100 | 0.8826 | 0.6458 | 0.6940 |

完整机器可读结果见 `results/bm25_scifact_dev.json`。Recall 按每条 query 的相关文档覆盖比例计算；MRR 使用首个相关文档排名，因此多相关文档 query 上 `Recall@1` 与 `MRR@1` 不必相等。

## 质量优化实验（同一 SciFact Dev）

训练只使用 809 条 train claim，开发集固定为 300 条 claim，避免用 Dev 调参。双塔训练采用 BM25 Top-100 难负样本（6,472 个，排除 703 次标注正例）和 mean-pooling InfoNCE；结果如下：

| 方法 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 78.74% | 64.16% | 67.16% |
| Zero-shot MiniLM Dense | 77.43% | 57.84% | 62.39% |
| Fine-tuned Dense (hard negative) | 80.32% | 63.86% | 67.54% |
| BM25 + fine-tuned Dense (fixed RRF) | **84.06%** | **68.97%** | **71.95%** |
| Query-adaptive RRF（规则） | 83.49% | 68.27% | 71.31% |
| RRF + 通用 Cross-Encoder Top-50 | 80.66% | 65.09% | 68.23% |

相对 BM25，固定 RRF 的 Recall@10、MRR@10、nDCG@10 分别提升 5.32、4.81、4.79 个百分点。规则自适应融合和通用重排的退化说明：权重规则没有在训练集上学习，通用 MS-MARCO 重排器与科学证据排序目标不一致；因此当前版本选择固定 RRF，并把领域 Cross-Encoder 微调列为下一步，而不是只保留最好看的结果。完整 run、配置和负结果见 `results/quality_ablation/quality_report.json`。

同一 BM25 配置也已在 809 条训练 claim 上完成 hard-negative mining：每条保留 8 个负例，共写出 809 条训练记录，并在候选池中排除了 703 次已标注正例命中。该次运行没有启用 teacher model，所以 `removed_teacher_ambiguous=0`；这不是“数据中不存在假负例”的结论。统计见 `results/scifact_train_bm25_negatives.jsonl.stats.json`，完整训练 JSONL 默认不提交 Git。

## 算法链路

```text
SciFact corpus / claims
        |
        +--> BM25 baseline -----------------------> Recall / MRR / nDCG
        |
        +--> top-ranked non-gold documents
                    |
                    +--> labeled-positive removal
                    +--> optional teacher ambiguity filtering
                    |
                    +--> query / positive / hard negatives
                                   |
                                   +--> custom InfoNCE dual encoder
                                                   |
                                                   +--> dense top-K
                                                           |
                                                           +--> Cross-Encoder rerank
```

## 二次开发扩展

- `retrieval_lab/vector_store.py` 提供 Milvus 2.4+ 的可选后端，保留当前
  Qdrant local-mode 作为无服务依赖的复现实验后端；两者都使用 cosine
  向量检索和显式 collection schema。
- `retrieval_lab/knowledge_router.py` 提供 source-aware 的 Wikipedia 辅助
  通道。Wiki 只能作为背景知识源，返回结果带 `source="wikipedia"`，不会
  混入 SciFact gold evidence 或训练标签。
- 后续实验应把 Milvus、Wiki 通道、query rewrite/decomposition 和
  evidence-aware rerank 放进同一 validation-only 消融矩阵，并单独报告
  召回、证据支持率、拒答率和延迟。

本地实测 hard-negative MiniLM、5,183 文档、300 条 Dev、Top-10：Qdrant
和 Milvus Recall 均为 81.40%；Qdrant P95 12.74 ms、索引 20.46 MiB，
Milvus P95 12.16 ms、Docker volume 38.14 MiB。结果见
`results/vector_store_benchmark.json`。

领域 Cross-Encoder 使用 919 个正例和 2,427 个 BM25 难负例完成 1 epoch
训练。BM25 Top-10 内重排后 MRR@10 从 64.16% 提升到 67.03%，nDCG@10
从 67.16% 提升到 69.40%；Recall@10 不变，因为只重排已有候选。

### ColBERT / SPLADE / CRAG-style 真实模型消融

`scripts/run_real_model_ablation.py` 使用公开 `colbert-ir/colbertv2.0`
（含 128 维 projection、query/document marker、query mask augmentation 和
MaxSim）与 `naver/splade_v2_max` 的正式权重，对官方 SciFact Dev 全部 300
条 claim 的 BM25 Top-20 候选进行重排。结果不是 proxy：

| 方法 | Recall@10 | MRR@10 | nDCG@10 | CPU ms/query |
|---|---:|---:|---:|---:|
| BM25 candidate order | 78.74% | 64.16% | 67.16% | - |
| ColBERT v2 rerank | 74.31% | 60.17% | 62.97% | 1,542 |
| SPLADE v2 rerank | 76.59% | 63.19% | 65.70% | 1,936 |

两种通用 checkpoint 都没有超过 BM25；候选池相同，因此 Recall@20 均为
82.92%。完整配置和结果见 `results/real_model_ablation_dev300.json`。旧的
`results/frontier_ablation_proxy.json` 仅是确定性 lexical proxy，不能与此表混用。

`scripts/run_crag_retry.py` 进一步比较了真实领域 Cross-Encoder、hard-negative
dense retry 和 DeBERTa sentence NLI reflection。Dev 按 query ID 固定拆成 92
条 calibration 与 208 条 evaluation，阈值只在 calibration 上选择。evaluation
中条件路由重试 33 条（15.87%），Recall@10/MRR@10/nDCG@10 为
78.08%/66.77%/68.62%，低于不重试的 BM25+领域 CE 的
78.96%/67.23%/69.22%。所以该 CRAG-style 路由作为负结果保留，不升级为默认链路；
这也不是完整 CRAG 或 Self-RAG 训练复现。结果见 `results/crag_retry_dev300/metrics.json`。

InfoNCE 中，每个 query 的第 0 类是对应 positive，后续类别是该 query 的 hard negatives；开启 in-batch negatives 后，其他 query 的 positives 也作为负例。teacher filter 会删除与最强正例分数差小于 margin 的候选，以降低未标注相关文档被错误当作负例的风险。

## 快速开始

环境：Python 3.10+。先进入仓库并安装：

```bash
python -m pip install -e ".[dev]"
```

SciFact 官方数据目录应包含：

```text
corpus.jsonl
claims_train.jsonl
claims_dev.jsonl
claims_test.jsonl
```

示例配置假设从本仓库根目录运行，并将官方数据放在 `data/scifact`；也可以通过 `--data-root` 临时覆盖。

运行 BM25 开发集基线：

```bash
python -m retrieval_lab.cli bm25 --config configs/scifact_bm25.json
```

挖掘训练集 BM25 难负样本：

```bash
python -m retrieval_lab.cli mine-negatives --config configs/scifact_mine_negatives.json
```

如需教师过滤，在配置中设置 Cross-Encoder：

```json
{
  "teacher_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "teacher_margin": 0.1
}
```

训练双塔模型：

```bash
python -m retrieval_lab.cli train-dual --config configs/scifact_dual_encoder.json
```

评测训练 checkpoint：

```bash
python -m retrieval_lab.cli dense-eval \
  --data-root data/scifact \
  --claims-file claims_dev.jsonl \
  --checkpoint checkpoints/scifact-minilm-infonce \
  --output-metrics results/dense_scifact_dev.json
```

重排 BM25 或 dense run：

```bash
python -m retrieval_lab.cli rerank \
  --data-root data/scifact \
  --claims-file claims_dev.jsonl \
  --input-run results/bm25_scifact_dev.run.jsonl \
  --model-name-or-path cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank-depth 50 \
  --output-metrics results/rerank_scifact_dev.json
```

启动 HTTP benchmark：

```bash
python scripts/benchmark_service.py --data-root data/scifact \
  --endpoint /v1/rag --requests 100 --concurrency 8 \
  --output results/benchmark_rag_c8.json
python scripts/evaluate_service.py --data-root data/scifact \
  --top-k 10 --concurrency 8 \
  --output results/service_hybrid_scifact_dev.json
```

## 实验协议

建议固定 SciFact train/dev 划分，并按以下顺序记录真实结果：

| 实验 | 训练负样本 | 假负例处理 | 第二阶段 |
|---|---|---|---|
| BM25 | 无 | 无 | 无 |
| Zero-shot encoder | 无 | 无 | 无 |
| Random-negative dual encoder | 随机 | 仅排除 gold | 无 |
| BM25-HN dual encoder | BM25 top-K | 仅排除 gold | 无 |
| Filtered-HN dual encoder | BM25 或 dense top-K | gold + teacher margin | 无 |
| Retriever + Reranker | 最优训练策略 | 同上 | Cross-Encoder |

每次实验至少保存配置、随机种子、训练 loss、完整 run 和指标 JSON。不能只报告最好数字，还要检查：

- hard negatives 是否包含未标注但语义相关的 false negatives；
- Recall@100 上升而 nDCG@10 不升时，是否只是相关文档排位仍靠后；
- Reranker 提升 nDCG 但增加多少 P95 延迟；
- temperature、负样本个数和 teacher margin 是否造成过拟合或样本过度过滤。

## 测试

官方数据审计（只读取 train/dev/test，不用 test 做训练或阈值选择）：

```bash
python scripts/audit_scifact.py --data-root data/scifact \
  --output results/dataset_audit.json
```

当前审计记录了 5,183 篇摘要、809/300/300 条 train/dev/test claim、证据句
标注数、支持/反驳分布和跨 split 重复。SciFact 原始标注没有 NEI claim
标签，不能把未标注 claim 直接计作 NEI；NEI 只能在明确的任务构造中单独
定义并报告。

无需 pytest 也可运行标准库测试：

```bash
python -m unittest discover -s tests -v
```

测试覆盖 BM25 排序、指标计算、假负例过滤、自定义 InfoNCE、padding-aware mean pooling 和 rerank 接口。

## 目录

```text
retrieval_lab/
  bm25.py       BM25 和倒排索引
  data.py       SciFact 数据模型与 JSONL I/O
  metrics.py    run 文件及 Recall/MRR/nDCG
  negatives.py hard-negative mining 与假负例过滤
  neural.py     Transformer 双塔、自定义 InfoNCE、dense retrieval
  rerank.py     PairScorer 协议和 Cross-Encoder
  serving.py    Embedding、Qdrant、混合召回、Rerank 和 FastAPI RAG
  training.py   triplet dataset、collator 和训练循环
  cli.py        端到端命令行
```

## 数据与上游说明

- SciFact claims/evidence annotations: CC BY 4.0。
- SciFact abstracts 来自 S2ORC: ODC-By 1.0。
- 本仓库不提交 SciFact 原始数据和模型权重。
- hard-negative 和 teacher-score 实验设计参考了 [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) 的公开工作流。FlagEmbedding 使用 MIT License；本仓库没有复制其完整仓库或源文件，而是独立实现了适合 SciFact 小规模实验的接口。
- Transformer 加载和模型配置依赖 Hugging Face `transformers`。

本仓库代码使用 MIT License。引用或修改上游项目时应继续保留其各自许可证与署名。
