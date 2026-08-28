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

尚未完成或不能声称的结果：

- 当前没有提交神经双塔或 Reranker 的训练后指标。
- 当前没有声称优于 BM25、BGE、SciBERT 或论文结果。
- 配置中的预训练模型只是可运行起点，不代表已经完成对应 GPU 实验。

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
