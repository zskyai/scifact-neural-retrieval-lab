# SciFact Neural Retrieval Lab

面向科学事实核验的可训练检索系统：从 BM25 基线出发，训练领域双塔检索器，使用混合召回、Cross-Encoder 重排和句子级证据选择，最后通过自适应路由决定回答、重试或 NEI。

项目重点不是把摘要塞进向量库，而是把 **hard negative、训练目标、候选排序、证据定位和服务延迟** 放进同一套可复现实验链路。

## 结果概览

官方 SciFact corpus 包含 5,183 篇摘要；训练集 809 条 claim，开发集 300 条 claim。以下结果均来自同一官方 dev split：

| 方法 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 78.74% | 64.16% | 67.16% |
| Zero-shot MiniLM Dense | 77.43% | 57.84% | 62.39% |
| Fine-tuned Dense（hard negative） | **81.40%** | **67.10%** | **70.15%** |
| BM25 + Dense fixed RRF | **84.06%** | **68.97%** | **71.95%** |
| BM25 + domain Cross-Encoder（Top-10） | 78.74% | **67.03%** | **69.40%** |
| Hybrid RRF + generic Cross-Encoder/evidence rerank | 84.06% | 65.07% | 69.14% |

领域 Cross-Encoder 的 `67.03%/69.40%` 来自 BM25 Top-10 重排；混合 RRF 后再用通用 Cross-Encoder 和句子证据特征重排的结果是 `65.07%/69.14%`。两组使用不同候选 run，分别列出。Fine-tuned Dense 的指标来自 `results/dense_hn_scifact_dev_recheck.json`，向量库延迟与索引大小来自 `results/vector_store_benchmark.json`。

服务侧对 5,183 篇摘要建立向量索引：

| Backend | Recall@10 | P95 | Index size |
|---|---:|---:|---:|
| Qdrant local | 81.40% | 12.74 ms | 20.46 MiB |
| Milvus | 81.40% | 12.16 ms | 38.14 MiB |

固定 100 请求、预热 10、并发 8 的端到端 benchmark：`/v1/retrieve` 为 59.17 QPS、P95 162.80 ms；`/v1/rag` 为 2.61 QPS、P95 3.63 s。

## 证据核验与数据优化（2026-11-08）

在官方 dev 全部 300 条 claim 上，领域 verifier 的 Accuracy 为 59.33%、Macro-F1 为 54.90%。固定模型、检索候选和标签决策，仅将单句引用改为训练内校准的多句证据集合，得到：

| 引用策略 | Evidence Precision | Evidence Recall | Evidence F1 | 过度拒答率 |
|---|---:|---:|---:|---:|
| 单句引用 | 47.50% | 20.77% | 28.90% | 27.66% |
| 多句证据集合 | 38.30% | 40.71% | 39.47% | 27.66% |

证据指标按标注句并集计算；两组标签决策一致。多句策略增加召回，也增加无关引用。结果与逐条预测见 `results/domain_verifier_v1_evidence_set_dev300/`，成对比较见 `results/domain_verifier_v1_evidence_set_comparison_v2/`。这是缓存推理上的引用策略消融，沿用原始推理耗时；引用语义支持率尚未人工核验。

训练数据审计定位到两个问题：旧版 256-token 摘要输入会截断标注证据；训练与校准的 cited papers 虽然隔离，hard negatives 仍有论文交集。新增 `retrieval_lab/evidence_training.py` 对完整 rationale set 编码，不把联合证据中的每个句子单独标为充分证据，也不把未标注句直接判为 NEI；候选论文按分区整体分配，覆盖正样本和负样本来源。

对全部 809 条 official-train claim 执行的新审计保留全部 957 组完整证据：676 条训练 claim 对应 801 组，133 条校准 claim 对应 156 组，均可放入 256 tokens，候选论文分区交集为 0。该产物是下一版训练的数据基础，旧版 checkpoint 未被覆盖。

```bash
python scripts/prepare_complete_rationales.py \
  --tokenizer checkpoints/scifact-stance-verifier-v1 \
  --output results/complete_rationale_audit_new
```

`train_domain_verifier.py` 已接入完整摘要窗口、CONTRADICTS 专项 hard negatives 和类别权重。最佳 checkpoint 在 133 条训练内 calibration claims 上选择：pair Macro-F1 64.49%、CONTRADICTS F1 54.37%；随后冻结配置评测官方 dev。运行清单为 `results/contradict_hard_negative_verifier_v2/run_manifest.json`，完整摘要特征为 `results/dev_full_abstract_features_v1/manifest.json`。

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

## MultiVerS 联合训练

`retrieval_lab/multivers_joint.py` 实现 Longformer 的 stance+rationale membership 联合模型：输入按完整句子切窗，完整 rationale set 才产生 sentence membership 监督，缺少完整 rationale 的窗口自动 mask。训练入口支持 10/20 hard negatives、512/1024/4096 窗口、假负例过滤和 membership/complete 监督策略：

```bash
python scripts/train_multivers_joint.py --model ../../models/allenai__longformer-base-4096 --output results/multivers_joint_neg10_len1024_v2 --negatives 10 --max-length 1024 --resume
```

`scripts/evaluate_multivers_joint.py` 在训练内 calibration claims 选择阈值，再在官方 dev300 输出标签、证据、拒答和时延；`scripts/score_official_scifact.py` 使用固定官方 evaluator。`scripts/run_multivers_ablation.py` 保存每组配置和 preflight，未完成的训练不会产生 complete 结果。

### 特征融合与选择性拒答

`retrieval_lab/evidence_fusion.py` 将摘要 stance、句子 stance margin、证据覆盖、支持/反驳冲突、claim-句子词项覆盖、检索 rank 和句子集中度联合成 14 维特征。`FeatureFusionCalibrator` 在 official-train 的 133 条校准 claim 上拟合多类模型，并在同一校准集选择 label-specific refusal policy；推理输出 `no_answerable_evidence`、`weak_evidence`、`conflicting_evidence` 和 `low_evidence_quality` 原因。`select_evidence_set` 用 coverage gain、重复惩罚和冲突惩罚选择最多 3 条真实句子，citation 位置不会脱离检索候选。

```bash
python scripts/run_feature_fusion_calibration.py \
  --calibration-scores results/domain_verifier_v1_bm25_calibration/scores.jsonl \
  --claims-train data/scifact/claims_train.jsonl \
  --dev-predictions results/domain_verifier_v1_bm25_dev300/predictions.jsonl \
  --baseline-summary results/domain_verifier_v1_bm25_dev300/summary.json \
  --output results/feature_fusion_calibration_v4
```

该 replay 的作用是验证新特征是否改善质量—覆盖率权衡，不替代正式训练。修正 stance conflict 特征后，当前 official dev replay 为 Accuracy 58.33%、Macro-F1 52.56%、Evidence F1 25.65%、unsafe answer 15.18%、过度拒答 30.32%；相对旧决策 Macro-F1 54.90% 和过度拒答 27.66%，说明新流程目前只降低了不安全回答，尚未形成总体提升，仍需针对 CONTRADICTS 和完整摘要上下文继续训练。完整 scorer 输出在 `results/feature_fusion_calibration_v7/official_score/summary.json`，特征分布审计在 `results/feature_fusion_calibration_v6/feature_audit.json`。

| 模块 | 作用 |
|---|---|
| `retrieval_lab/bm25.py` | 倒排索引、BM25 打分和批量检索 |
| `retrieval_lab/neural.py` | Transformer 双塔、mean pooling、InfoNCE 和 dense retrieval |
| `retrieval_lab/negatives.py` | BM25 hard-negative mining、gold 排除和 teacher margin 过滤 |
| `retrieval_lab/rerank.py` | Cross-Encoder pair scoring 与候选重排 |
| `retrieval_lab/query_planning.py` | 可审计 query rewrite/decomposition 与 provenance |
| `retrieval_lab/adaptive_routing.py` | 基于证据覆盖和冲突特征的 answer/retry/abstain 路由 |
| `retrieval_lab/evidence_agent.py` | 有界检索 agent：多查询召回、证据核验、定向重试和 NEI 决策 |
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

仓库提供面向科研审阅者的 claim review desk：研究者提交待核验 claim，系统对 SciFact 摘要执行 hybrid retrieval、句级 NLI 和有界 retry，保存判定、证据与完整 trace；审阅者随后录入 SUPPORTS / CONTRADICTS / NEI 裁决及理由。案件和人工裁决写入本地 SQLite，供复核与后续误差分析，不会自动混入训练集。

```bash
python scripts/serve_rag.py \
  --data-root data/scifact \
  --qdrant-path results/qdrant_hn_local \
  --collection scifact_hn \
  --embedding-model checkpoints/scifact-minilm-hn \
  --rerank-model checkpoints/scifact-cross-encoder-domain \
  --sentence-nli-model ../../models/cross-encoder__nli-deberta-v3-base
```

打开 `http://127.0.0.1:8000/`。`POST /v1/cases` 创建持久化核验案件；`GET /v1/cases` 查询审阅队列；`POST /v1/cases/{case_id}/reviews` 保存人工标签。证据不足时进入 needs-review 状态。界面提供完整摘要、逐句 NLI、执行 trace、人工裁决历史和 JSON 报告下载。

应用链路接入领域 hard-negative encoder、领域二分类 Cross-Encoder 和句级 DeBERTa NLI。服务按二分类 logit margin 重排，每个候选文档对应一个分数；纯反驳证据可以输出 CONTRADICTS，支持和反驳同时出现时触发冲突路由。当前路由阈值是固定启发式配置。

全部 **300 条官方 dev claim** 已通过应用接口完成核验，失败 0。结果见 `results/review_application_dev300_scored_2026-11-07/summary.json`，逐条预测和错误队列保存在同目录。

| 应用端到端指标 | 结果 |
|---|---:|
| Label Accuracy / Claim Macro-F1 | 44.67% / 44.23% |
| Evidence Precision / Recall / F1（micro） | 12.01% / 17.49% / 14.24% |
| Citation ID Validity | 100.00% |
| NEI 拒答准确率 / 过度拒答率 | 49.11% / 41.49% |
| 服务计算耗时 P50 / P95 | 1.89 / 4.58 s |

证据按标注句子并集匹配，非官方 joint evidence-set scorer。Citation ID 有效只表示文档和句号真实存在，语义 support 尚未独立人工评分。该结果说明可追溯链路已接通，同时暴露 NLI 泛化与路由误拒答问题；不能用上面的检索 Recall 替代核验质量。

### 领域核验器与拒答校准

新增三分类 MiniLM 领域核验器，在官方 train 中划出 676 条训练 claim 和 133 条校准 claim。6,660 个训练样本覆盖摘要、标注理由集合、证据句和 BM25 难负例，完成四轮训练；checkpoint 按留出组的 pair Macro-F1 选择。核验时对 BM25 Top-10 的摘要和全部句子打分，按文档联合摘要与句子置信分数，只在真实来源位置上输出引用。

拒答阈值只在 133 条留出训练 claim 上选择：threshold=0.25、stance margin=0.03、support bias=0.05。冻结配置后完成全部 300 条官方 dev，失败 0：

| 官方 dev 核验指标 | 原 hybrid + 通用 NLI 链路 | BM25 + 领域核验器 + 校准 |
|---|---:|---:|
| Label Accuracy | 44.67% | **59.33%** |
| Claim Macro-F1 | 44.23% | **54.90%** |
| Evidence Precision / Recall / F1 | 12.01% / 17.49% / 14.24% | **47.50% / 20.77% / 28.90%** |
| NEI 判定准确率 | 49.11% | **78.57%** |
| 过度拒答率（应回答却 NEI） | 41.49% | **27.66%** |
| NEI 误答率（应 NEI 却回答） | 50.89% | **21.43%** |
| Citation ID Validity | 100.00% | 100.00% |

逐 claim 配对比较修正 102 条、退化 58 条，净增加 44 条正确判定。Accuracy 增量为 14.67 个百分点，10,000 次配对 bootstrap 的 95% 区间为 [6.67, 22.67] 个百分点。两条链路的检索、核验和证据选择均有变化，这张表衡量系统整体变化。新链路 SUPPORTS / CONTRADICTS / NEI 的 F1 分别为 60.09% / 34.78% / 69.84%。其中 CONTRADICTS F1 从旧版的 50.78% 降到 34.78%，总体改善伴随反驳识别退化；多句证据召回也仍偏低。

结果与配置：`results/domain_verifier_v1_bm25_dev300/summary.json`、`results/domain_verifier_v1_paired_comparison/comparison.json`、`results/domain_verifier_v1_bm25_calibration/calibration.json`。完整预测、配对记录和错误队列保存在对应目录。旧服务 P50/P95 为 1.89/4.58 秒，新引擎为 2.54/3.52 秒；运行入口和计算负载不同，不能据此推导单组件加速。

数据审计保存在 `results/domain_verifier_v1_bm25_calibration/source_audit_v2.json`：训练与校准 claim、标注引用论文均无交集，但 26 篇校准论文曾作为其他训练 claim 的难负例出现，共 78 个样本，因此该版本不是论文文本完全隔离的实验。校准组也用于选择 checkpoint。事后审计发现实际训练 claim 与官方 dev 存在一条文本重复（train 1291 / dev 1292）；排除该 dev claim 后，剩余 299 条的 Accuracy 为 59.20%、Macro-F1 为 54.82%，见 `results/domain_verifier_v1_paired_comparison/exact_duplicate_sensitivity.json`。test 未参与训练或选择。证据指标使用标注句并集匹配，语义引用支持和人工一致性尚未评分。

```bash
python scripts/evaluate_review_application.py \
  --domain-checkpoint checkpoints/scifact-stance-verifier-v1 \
  --calibration results/domain_verifier_v1_bm25_calibration/calibration.json \
  --output results/domain_verifier_new_dev300 --max-retries 0 --threads 4
```

应用服务可通过 `--domain-verifier checkpoints/scifact-stance-verifier-v1 --verifier-calibration results/domain_verifier_v1_bm25_calibration/calibration.json` 启用同一核验引擎。该模式要求 Top-10、零重试；界面读取服务配置并固定参数，API 拒绝偏离校准配置的请求。检索、重排接口仍可独立调用。

```bash
python scripts/evaluate_review_application.py \
  --data-root data/scifact \
  --output results/review_application_new
python scripts/export_review_candidates.py \
  --db results/research_cases.sqlite3 \
  --output results/review_candidates/cases.jsonl
```

导出的人工裁决保留模型输出、复核人和理由，用于误差分析。进入训练前需要确认 split 和人工证据句；验证或测试案件不会自动回流训练。

使用 `python scripts/import_evaluation_reviews.py --predictions results/review_application_dev300_scored_2026-11-07/predictions.jsonl` 将本次 180 条带错误标记的案件导入复核队列，保留 gold label、错误类别、split 和 run hash。重复执行跳过已有记录，不会生成虚假的人工裁决。

启动服务后提供：

| Endpoint | 作用 |
|---|---|
| `GET /healthz` | 服务状态 |
| `POST /v1/embed` | 文本向量化 |
| `POST /v1/plan` | 返回 rewrite、subquery 和 provenance |
| `POST /v1/route` | 返回 answer/retry/abstain 决策与特征 |
| `POST /v1/agent/verify` | 执行多步检索、句级核验、重试并返回完整 trace |
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
  --qdrant-path results/qdrant_local \
  --sentence-nli-model ../models/deberta-v3-scifact-nli
```

调用证据核验 agent：

```bash
curl -X POST http://127.0.0.1:8000/v1/agent/verify \
  -H "Content-Type: application/json" \
  -d '{"claim":"The intervention reduces mortality","top_k":10,"max_retries":1}'
```

响应包含 `label`、`confidence`、证据句、每轮检索工具调用和路由原因。启动时可传本地句级 NLI checkpoint；未传时使用 lexical fallback。

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
