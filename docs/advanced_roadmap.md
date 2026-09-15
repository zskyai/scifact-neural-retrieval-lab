# Advanced Development Roadmap

All changes use SciFact train for optimization, dev for selection, and keep test
sealed until the final run.

1. Retrieval: compare BM25, domain dual encoder, SPLADE sparse expansion and
   ColBERT late interaction. Store dense/late-interaction vectors in Milvus and
   report recall, index size, build time and P95 latency.
2. Ranking: train a SciBERT/DeBERTa cross-encoder on claim-document and
   claim-sentence pairs. Compare abstract, sentence and multi-granularity input.
3. Dynamic retrieval: add RAG-Fusion/HyDE rewrites only when first-pass score
   margin or evidence coverage is low. Measure quality gains against extra token
   and latency cost.
4. Evidence verification: run NLI over selected sentences, expose support,
   contradiction and conflict features, and calibrate NEI on dev only.
5. External knowledge: route Wikipedia as a separately tagged background source.
   It may improve terminology recall but never counts as SciFact gold evidence.
6. Reliability: use temperature scaling or conformal/selective prediction for
   confidence and report coverage-risk curves, false refusal and hallucinated
   citation rates.

Promotion rule: a component is retained only when it improves evidence F1 or
claim Macro-F1 without an unacceptable increase in false refusal, P95 latency,
index footprint or token cost.
