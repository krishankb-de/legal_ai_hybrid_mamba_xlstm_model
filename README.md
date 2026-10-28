# lexhybrid

[![CI](https://github.com/krishankb-de/legal_ai_hybrid_mamba_xlstm_model/actions/workflows/ci.yml/badge.svg)](https://github.com/krishankb-de/legal_ai_hybrid_mamba_xlstm_model/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue) ![License: proprietary](https://img.shields.io/badge/license-proprietary-lightgrey)

`lexhybrid` is a citation-faithful, retrieval-gated language model for legal questions. A hybrid
decoder (Mamba-3 state-space layers, mLSTM/xLSTM layers and sparse attention) reads a question
together with numbered retrieved passages and answers by emitting *pointers* into those passages. A
renderer substitutes the exact source sentence and its citation label, and a verifier decides whether
the answer may be shown at all.

Pretraining covers German law; Austrian, Swiss and EU sources are collected for retrieval. It is an
actively developed research codebase: no trained weights are released and there is no end-to-end
answer command yet, so the components below are used as a library.

## How it answers

The decoder sees the question and the retrieved passages, each sentence numbered:

```
<|question|>Wann darf der Vermieter kündigen?
<|passage|><|c1|>BGB §573
<|s1|>Der Vermieter kann nur kündigen, wenn er ein berechtigtes Interesse an der Beendigung des
Mietverhältnisses hat.<|s2|>Ein berechtigtes Interesse liegt insbesondere vor, wenn der Vermieter
die Räume als Wohnung für sich benötigt.<|/passage|>
<|answer|>
```

It never writes the statute text. It writes free text and pointers:

```
Er braucht ein berechtigtes Interesse: <|q|><|c1|><|s1|>
```

`<|q|><|c1|><|s1|>` means "quote sentence 1 of passage 1" and `<|cite|><|c1|>` means "cite passage 1".
The verifier checks the claim against the passage and the renderer produces what the reader sees. The
passage text below is illustrative, and the NLI model is a stand-in that always answers 0.99 (the
production model is `lexhybrid.verifier.nli.MDebertaNLI`):

```python
from lexhybrid.retrieval.chunking import Chunk
from lexhybrid.verifier.verifier import verify

chunk = Chunk(
    id="gii:bgb:573#1",
    doc_id="gii:bgb:573",
    source="gii",
    jurisdiction="DE",
    doc_type="statute",
    citation_id="BGB §573",
    hierarchy=("BGB", "§573"),
    sentences=(
        "Der Vermieter kann nur kündigen, wenn er ein berechtigtes Interesse an der Beendigung "
        "des Mietverhältnisses hat.",
        "Ein berechtigtes Interesse liegt insbesondere vor, wenn der Vermieter die Räume als "
        "Wohnung für sich benötigt.",
    ),
    labels=("BGB §573 Abs. 1 S. 1", "BGB §573 Abs. 2 Nr. 2"),
    valid_from=None,
    valid_to=None,
    licence="DE-UrhG-5",
    commercial_safe=True,
    research_only=False,
    url="https://www.gesetze-im-internet.de/bgb/__573.html",
)
answer = "Er braucht ein berechtigtes Interesse: <|q|><|c1|><|s1|>"

result = verify("Wann darf der Vermieter kündigen?", answer, [chunk], lambda premise, hypothesis: 0.99)
print(result.rendered)
```

Output:

```
Er braucht ein berechtigtes Interesse: „Der Vermieter kann nur kündigen, wenn er ein berechtigtes Interesse an der Beendigung des Mietverhältnisses hat.“ (BGB §573 Abs. 1 S. 1)

Hinweis: Dies ist eine rechtliche Information, keine Rechtsberatung.
```

A pointer to a passage or sentence that does not exist, a quote that differs from its source, or a
claim the cited passage does not entail makes the verifier refuse instead.

## Key properties

- **Pointer citations.** Constrained decoding makes a pointer to a nonexistent passage or sentence
  impossible; the model never generates statute text.
- **Exact quotes and labels.** The renderer inserts the passage's own sentence, byte for byte, with
  its citation label, for example `BGB §573 Abs. 2 Nr. 1`.
- **Verifier gate.** Before an answer is shown, every pointer must resolve, every quote must match
  its source, every claim must be entailed (NLI) by the passage it cites, and questions asking for
  the identity of a party are refused.
- **Notice on every answer.** Every answer, a refusal included, ends with the *Hinweis* line.
- **Retrieval filters.** Jurisdiction (`DE`, `AT`, `CH`, `EU`), document type and valid-on-date
  filters, BM25 with German stemming, rank fusion and cross-encoder reranking.
- **Licence-aware corpus.** A licence register and a build check keep research-only sources out of
  the commercial-safe training arm; personal names are scrubbed before training.

## Results at a glance

| Measurement | Result | Report |
|---|---|---|
| Base model size | 220.8M parameters (104.1M excluding embeddings), 12 layers: 7 Mamba-3, 3 mLSTM, 2 attention | [param_counts.md](analysis/param_counts.md) |
| Tokenizer cost on DACH legal text | Qwen3 2.280 tokens/word against 2.872 for GPT-2 (21% fewer tokens) | [tokenizer_fertility.md](analysis/tokenizer_fertility.md) |
| Commercial-safe pretraining corpus | 6,092,655,089 training tokens after scrub and deduplication; the pretraining budget is 5 B tokens | [corpus_manifest.md](analysis/corpus_manifest.md) |
| Architecture screen | Base hybrid: two-seed mean validation PPL 5.3582 (seeds 42 and 43); no variant beats it by more than the 0.10 PPL bar; removing attention costs 0.675 PPL | [screen_results.md](analysis/screen_results.md) |
| Distillation teacher memory | Qwen3-8B-Base with gradient checkpointing: 43.18 GB peak allocated per GPU at 4,096-token rows (out of memory without it) | [profile_results.md](analysis/profile_results.md) |

These are small-scale measurements of the training and retrieval stack, not evidence of
legal-reasoning quality.

## Status

Implemented: the hybrid model and its kernels, pointer-constrained decoding, the pretraining and
distillation entry point, the corpus collectors, scrub, deduplication and packing, evaluation sets and
probes, BM25 retrieval with filters, fusion and reranking, the renderer and the verifier, and the
tokenizer with its 87 special tokens (pointers and prompt markers).

Not yet available: a dense index builder, an end-to-end answer command, supervised fine-tuning on
pointer answers, and released weights.

## Architecture at a glance

```
question -> retrieve + rerank -> prompt (numbered passages) -> hybrid decoder (pointer-constrained)
         -> pointer answer -> verifier -> renderer (exact sentence + label) -> answer + notice
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the model, decoding, training, retrieval,
renderer, verifier and data pipeline.

A generated knowledge graph of the code (modules, classes, functions and their call and import
relations) is in [`graphify-out/`](graphify-out/GRAPH_REPORT.md). Regenerate it with
[graphify](https://pypi.org/project/graphifyy/) (`graphify update .`) after code changes.

## Repository layout

| Path | Contents |
|---|---|
| `lexhybrid/` | the package: `config`, `layers`, `kernels` (SSD, TFLA, selective scan), `models`, `decoding`, `training`, `data` (collectors, scrub, dedup, packing, eval sets, probes), `retrieval`, `renderer`, `verifier`, `eval`, `utils` |
| `configs/` | Hydra configs: `model/`, `trainer/`, `dataset/`, `callbacks/`, `distill/` |
| `scripts/` | training, evaluation, index/shard building, measurement and check scripts; `validate.sh` |
| `scripts/slurm/` | SLURM wrappers for cluster jobs and `cluster.env.example` |
| `tests/` | the pytest suite and its fixtures |
| `docs/` | [architecture](docs/ARCHITECTURE.md), [document schema](docs/DOCUMENT_SCHEMA.md), [corpus licence register](docs/CORPUS_LICENCE_REGISTER.md) |
| `analysis/` | measurement reports (see Results) |
| `data/manifests/` | per-source document manifests (no document text); collected data, shards and indexes stay local and are gitignored |
| `envs/scrub/` | the separate environment for the named-entity scrub |
| `graphify-out/` | generated code knowledge graph: [`GRAPH_REPORT.md`](graphify-out/GRAPH_REPORT.md) (communities, hub nodes), `graph.json`, and `graph.html` (interactive viewer, open locally) |

## Requirements

- Python 3.11 (`.python-version`)
- [uv](https://docs.astral.sh/uv/)
- torch 2.11.0, pinned in `uv.lock`: CPU/MPS wheel on macOS arm64, CUDA 12.8 wheel on Linux x86_64

`pyproject.toml` resolves the lock for exactly those two platforms (`[tool.uv] environments`).

## Installation

```bash
uv sync --locked                       # builds .venv exactly as uv.lock says
.venv/bin/python scripts/check_env.py  # verifies the environment equals the lock
```

## Quickstart

CPU smoke test: a forward and backward pass through every mixer type, checking that every
parameter receives a gradient.

```bash
.venv/bin/python scripts/smoke_model.py
```

Tests:

```bash
.venv/bin/python -m pytest tests/ -m "not slow and not cuda and not multigpu and not network" -q
```

Validation (environment, lint, config checks, tests, smoke, repository hygiene):

```bash
bash scripts/validate.sh          # fast selection (no slow/cuda/multigpu/network tests)
bash scripts/validate.sh --full   # adds the slow tests
bash scripts/validate.sh --ci     # what CI runs: --full plus coverage floor and JUnit output
```

## Training

`scripts/train_pretrain.py` is a Hydra entry point; `configs/config.yaml` composes a model, dataset,
trainer, callbacks and optional distillation config. Override any of them on the command line:

```bash
# CPU debug run on synthetic data (two optimizer steps)
.venv/bin/python scripts/train_pretrain.py model=hybrid_legal_base dataset=synthetic trainer=cpu_debug

# GPU run on the packed mixture, with logit distillation from a Qwen3-8B teacher
# (the 8B teacher needs gradient checkpointing to fit in memory)
.venv/bin/python scripts/train_pretrain.py model=hybrid_legal_base dataset=mixture_pretrain \
    trainer=h100_single_gpu distill=qwen3_8b model.use_gradient_checkpointing=true
```

Model, trainer and distillation configs are in `configs/model/`, `configs/trainer/`
(`cpu_debug`, `h100_single_gpu`, `h100_multi_ddp`) and `configs/distill/`. Model fields can be
overridden as `model.<field>=...`. `scripts/check_configs.py` composes every model config and checks
its operator pins.

## Data pipeline and retrieval index

Documents are collected into a common schema ([docs/DOCUMENT_SCHEMA.md](docs/DOCUMENT_SCHEMA.md)),
scrubbed, deduplicated and packed into shards. Pretraining uses German law only (`gii`, `rii`, `oldp`,
`dip`, plus general German text); Austrian, Swiss and EU sources are collected for retrieval. Licences are tracked in
[docs/CORPUS_LICENCE_REGISTER.md](docs/CORPUS_LICENCE_REGISTER.md).

```bash
# collect (each source has a collector module)
.venv/bin/python -m lexhybrid.data.corpus.collectors.gii --limit 20 --out data/raw/gii

# scrub (needs the separate environment: uv sync --locked --project envs/scrub)
.venv/bin/python -m lexhybrid.data.corpus.scrub_ler --source gii

# pack deduplicated, scrubbed documents into training shards (smoke scale: 20 documents)
.venv/bin/python scripts/build_shards.py --root data/shards/smoke --row-len 4096 \
    --val-fraction 0.1 --min-val-docs 2 --sources gii

# build a BM25 retrieval index from scrubbed documents
.venv/bin/python scripts/build_index.py --kind bm25 --out data/index/smoke
```

Run each script with `--help` for its options. The index is written under `data/index/`, which is
gitignored. Only the BM25 index is implemented.

## Cluster usage

Cluster jobs are SLURM wrappers in `scripts/slurm/` (environment setup, corpus build, scrub, pack,
training on 1 or 4 GPUs, GPU tests, profiling, probes). Copy `scripts/slurm/cluster.env.example` to
`scripts/slurm/cluster.env`, set the host alias, the repository path and the scratch root, and adapt
the `#SBATCH` headers (partition, account, constraints) to your site. The wrappers' partition and
account target one site and are asserted by `tests/test_slurm_wrappers.py`, so adapting them to
another cluster means changing both. Then:

```bash
source scripts/slurm/cluster.env
bash scripts/sync_to_cluster.sh        # rsync the working tree to the cluster
ssh "$CLUSTER_HOST" "cd $CLUSTER_REPO && sbatch scripts/slurm/gpu_tests.sh"
bash scripts/sync_from_cluster.sh logs # pull logs into cluster/logs/
```

Each wrapper documents its environment variables in its header.

## Limitations and intended use

- Research code under active development; no released weights. Results in `analysis/` are from
  small-scale runs and should not be read as evidence of legal-reasoning quality relative to other
  models.
- The system is designed for **legal information, not legal advice**. It answers only from the
  retrieved passages, refuses when they do not support an answer, and appends a notice to every
  answer. It does not replace a qualified lawyer.
- The model is pretrained on German law only. Austrian, Swiss and EU sources are available to
  retrieval but not to pretraining; retrieval quality bounds answer quality.
- Corpus licences differ per source; research-only sources never enter the commercial-safe arm.

## Acknowledgements and data attribution

- **Mamba-3** and **xLSTM** (mLSTM layers, with TFLA chunked kernels) are the published architectures
  this model builds on; the kernels here are plain PyTorch.
- **Qwen3** supplies the tokenizer and the distillation teachers (Apache-2.0).
- Corpus sources and their required attribution notices, among them Open Legal Data (ODbL 1.0),
  the CC BY 4.0 sources (EUR-Lex, RIS), the Bundestag DIP terms of use and entscheidsuche.ch, are listed
  in [docs/CORPUS_LICENCE_REGISTER.md](docs/CORPUS_LICENCE_REGISTER.md), which also lists the
  models the pipeline runs and the evaluation sets.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Proprietary. Copyright (c) 2026 The lexhybrid authors. All rights reserved. See [LICENSE](LICENSE).
