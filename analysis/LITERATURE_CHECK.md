# Prior-work check

## Purpose

This file records a search for published work close to `lexhybrid`, so that its documentation cites
what it builds on and claims novelty only where the search supports it. `lexhybrid` is a 220.8M-parameter
decoder-only model for German law. Its 12 layers are 7 Mamba-3, 3 mLSTM and 2 causal softmax-attention
layers, interleaved layer by layer. It is trained from scratch with online logit distillation from
Qwen3-8B-Base. It answers from retrieved passages (BM25, bge-m3 dense vectors, reciprocal rank fusion,
a bge-reranker-v2-m3 cross-encoder) by emitting pointer tokens (`<|q|><|cK|><|sJ|>` quotes sentence J
of passage K, `<|cite|><|cK|>` cites passage K). A finite-state logit mask makes invalid pointers
impossible to decode, a renderer substitutes the exact source sentence and its citation label, and
a verifier (pointer resolution, verbatim quote check, mDeBERTa NLI entailment at threshold 0.5,
refusal of party-identity questions and person names) removes the answer or refuses. Sources:
`README.md`, `docs/ARCHITECTURE.md`, `lexhybrid/verifier/nli.py`, `lexhybrid/retrieval/`.

## Method

- **Date of search:** 2026-11-03.
- **Tools:** a general web search engine (results restricted to arXiv, ACL Anthology and OpenReview
  for some queries) and direct retrieval of individual pages. Sources used: arXiv abstract and HTML
  pages, ACL Anthology, PMLR, NeurIPS and ICLR proceedings pages, Hugging Face model cards, Zenodo,
  dblp, PubMed Central, and vendor pages for commercial products.
- **Queries run** (grouped):
  - Architectures: "hybrid Mamba xLSTM attention language model"; "Mamba-3 state space model arXiv
    2026"; "xLSTM 7B recurrent LLM"; "mLSTM Mamba hybrid layers combined model"; "Mamba-xLSTM OR
    xLSTM-Mamba hybrid"; "xLSTM Mamba hybrid architecture three mixers ... interleaved"; "language
    model hybrid mLSTM Mamba-2 attention layers same model ablation layer pattern"; "xLSTM Mamba
    hybrid language modeling interleaved blocks" (arXiv/ACL/OpenReview only); "Mamba-3 hybrid with
    attention layers"; plus title queries for Jamba, Zamba, Samba, Griffin/RecurrentGemma, Hymba,
    Nemotron-H, Falcon-H1, Mamba-2, the NVIDIA Mamba-2-Hybrid study, Meta's hybrid design study,
    xLSTM, TFLA and "Mamba in the Llama".
  - Citation and quoting: title queries for Pointer Networks, CopyNet, pointer-generator networks,
    GopherCite, WebGPT, AttributedQA, ALCE, AGREE, Self-RAG, Attribute First then Generate, LongCite,
    Quote-Tuning, Copy Is All You Need, ReClaim; "constrained decoding citation identifiers retrieved
    passages sentence-level pointer tokens"; "verbatim RAG answer composed only of extracted spans";
    "LLM generates sentence identifiers to cite evidence sentences".
  - Verification: title queries for AIS, TRUE, RARR, AttrScore, MiniCheck, Magesh et al., Dahl et
    al., CitaLaw, LegalBench-RAG; "legal RAG citation verification NLI entailment ... hallucination
    detection"; "German legal AI product RAG citations ... hallucination evaluation Germany".
  - German legal NLP: GerLayQA, GerDaLIR, LegalQuAD, GerLeRB ("German Legislative Retrieval
    Benchmark"), German legal BERT models, German legal LLMs, German legal LLM benchmarks, RAG over
    German statutes, Noxtua, the LER NER dataset, Open Legal Data, MultiLegalPile, LEXTREME,
    "xLSTM legal text German law".
  - Small legal models and distillation: SaulLM-7B, LEGAL-BERT, Pile of Law, Gemma 2, MiniPLM,
    "pre-training distillation design space", "small legal language model trained from scratch
    knowledge distillation", "legal domain small decoder language model pretrained from scratch",
    "Mamba state space model legal domain".
- **How claims were checked.** Works marked † had their arXiv abstract page, HTML full text,
  proceedings page or record opened directly. The others were identified from search-result listings
  that showed the title, authors, venue and abstract text at the linked URL; their full text was not
  read. No statement below goes beyond what those pages said.
- **Limits.** English-language web search only; German-language legal-informatics venues (for
  example conference proceedings and journals published only in German) were not searched
  systematically. Unpublished or proprietary systems cannot be ruled out. A search finds what is
  indexed; "not found" below means not found by these queries on this date.

## 1. Hybrid state-space / recurrent / attention architectures

| Work | Authors, year | Venue / link | Shared with lexhybrid | Not shared |
|---|---|---|---|---|
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu, Dao, 2023 | [arXiv 2312.00752](https://arxiv.org/pdf/2312.00752) | selective SSM; the paper also tests interleaving Mamba with attention blocks as an ablation | Mamba-1 only; no mLSTM |
| Transformers are SSMs (Mamba-2, SSD) | Dao, Gu, 2024 | ICML 2024, [arXiv 2405.21060](https://arxiv.org/abs/2405.21060v1), [PMLR](https://proceedings.mlr.press/v235/dao24a.html) | the chunked SSD scan that `lexhybrid/kernels/ssd` implements | not a three-way hybrid |
| Mamba-3: Improved Sequence Modeling using State Space Principles † | Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu, 2026 | ICLR 2026, [arXiv 2603.15569](https://arxiv.org/abs/2603.15569) | the Mamba-3 layer lexhybrid uses | the abstract page reports no hybrid with attention or mLSTM; the body was not checked for hybrid experiments |
| Jamba: A Hybrid Transformer-Mamba Language Model | Lieber et al., 2024 | [arXiv 2403.19887](https://arxiv.org/pdf/2403.19887) | Mamba layers interleaved with a minority of attention layers | 52B total with MoE; Mamba-1; no mLSTM |
| Zamba: A Compact 7B SSM Hybrid Model | Glorioso et al., 2024 | [arXiv 2405.16712](https://arxiv.org/html/2405.16712v1) | Mamba backbone with periodic attention | one shared attention block; 7B; no mLSTM |
| Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling † | Ren, Liu, Lu, Shen, Liang, Chen, 2024 | ICLR 2025, [arXiv 2406.07522](https://arxiv.org/abs/2406.07522) | Mamba interleaved with attention | sliding-window attention; no mLSTM |
| Griffin: Mixing Gated Linear Recurrences with Local Attention | De, Smith et al., 2024 | [arXiv 2402.19427](https://arxiv.org/abs/2402.19427v1) | gated linear recurrence mixed with attention | RG-LRU recurrence, local attention; no SSM or mLSTM |
| RecurrentGemma | Google DeepMind, 2024 | [arXiv 2404.07839](https://arxiv.org/pdf/2404.07839) | released Griffin-architecture models (2B, 9B) | as Griffin |
| An Empirical Study of Mamba-based Language Models | Waleffe et al., 2024 | [arXiv 2406.07887](https://arxiv.org/abs/2406.07887v1) | Mamba-2 layers with a small share (7%) of attention layers, spread evenly | 8B; no mLSTM |
| Hymba: A Hybrid-head Architecture for Small Language Models | Dong et al., 2024 | ICLR 2025, [arXiv 2411.13676](https://arxiv.org/pdf/2411.13676) | small hybrid LM (1.5B) with SSM and attention | attention and SSM heads in parallel inside each layer; no mLSTM |
| Nemotron-H | NVIDIA, 2025 | [arXiv 2504.03624](https://arxiv.org/pdf/2504.03624) | Mamba-2 with about 8% attention layers | 8B to 56B; no mLSTM |
| Falcon-H1 | Zuo et al., 2025 | [arXiv 2507.22448](https://arxiv.org/abs/2507.22448) | SSM plus attention | attention and Mamba-2 heads in parallel within a block; no mLSTM |
| Hybrid Architectures for Language Models: Systematic Analysis and Design Insights | Bae et al., 2025 | [arXiv 2510.04800](https://arxiv.org/abs/2510.04800) | studies ratio and ordering of attention and SSM layers in inter-layer hybrids | no mLSTM |
| xLSTM: Extended Long Short-Term Memory | Beck, Pöppel et al., 2024 | NeurIPS 2024, [arXiv 2405.04517](https://arxiv.org/abs/2405.04517), [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract.html) | the mLSTM layer | mixes mLSTM with sLSTM, not with SSM or attention |
| xLSTM 7B: A Recurrent LLM for Fast and Efficient Inference | Beck et al., 2025 | ICML 2025, [arXiv 2503.13427](https://arxiv.org/pdf/2503.13427) | mLSTM-based LLM | 7B; no Mamba layers described |
| Tiled Flash Linear Attention | Beck, Pöppel, Lippe, Hochreiter, 2025 | NeurIPS 2025, [arXiv 2503.14376](https://arxiv.org/abs/2503.14376) | the chunked mLSTM kernel algorithm that `lexhybrid/kernels/tfla` reimplements in PyTorch | kernels, not a model |
| Short window attention enables long-term memorization (SWAX) † | Cabannes, Beck et al., 2025 | [arXiv 2509.24552](https://arxiv.org/abs/2509.24552) | xLSTM layers interleaved with attention layers | sliding-window attention; no Mamba |
| Effective Distillation to Hybrid xLSTM Architectures † | Hauzenberger, Schmidinger et al., 2026 | [arXiv 2603.15590](https://arxiv.org/abs/2603.15590) | distillation from Llama, Qwen and Olmo teachers into an mLSTM + sliding-window-attention hybrid | linearizes a pretrained teacher instead of training from scratch; no Mamba |
| On Subquadratic Architectures: From Applications to Principles † | Hartl, Zólyomi, Stap et al., 2026 | [arXiv 2606.12364](https://arxiv.org/html/2606.12364v1) | 400M inter-layer hybrids with 3 attention layers in 24; compares xLSTM, Mamba-2 and Gated DeltaNet as the recurrent operator | each hybrid uses one recurrent operator family; Mamba-2 and xLSTM never appear in the same model (the xLSTM hybrid mixes mLSTM, sLSTM and attention) |
| The Mamba in the Llama: Distilling and Accelerating Hybrid Models † | Wang, Paliotta, May, Rush, Dao, 2024 | NeurIPS 2024, [arXiv 2408.15237](https://arxiv.org/abs/2408.15237) | distillation of a Transformer teacher into Mamba/Mamba-2 + attention hybrids | reuses the teacher's attention projection weights; no mLSTM |
| When Mamba Meets xLSTM: xLSTM-VMUNet | Fang et al., 2024 | [arXiv 2411.09363](https://arxiv.org/html/2411.09363v3) | combines Mamba and xLSTM modules | image segmentation, not language modelling; no attention layers described |

**Finding.** Mamba + attention hybrids are well established (Jamba, Zamba, Samba, the Mamba-2-Hybrid
study, Nemotron-H, Hymba, Falcon-H1). xLSTM/mLSTM + attention hybrids are also published (SWAX,
Hauzenberger et al. 2026, Hartl et al. 2026). The only Mamba + xLSTM combination found is an image
segmentation model. No language model that puts Mamba (any version), mLSTM and softmax-attention layers
in one stack was found; the closest work (Hartl et al. 2026) compares these operators in separate
hybrids. A secondary topic listing of xLSTM hybrids
([Emergent Mind](https://www.emergentmind.com/topics/hybrid-xlstm-architectures) †) also lists no
xLSTM + SSM language model.

Note on wording: lexhybrid's "sparse attention" means two full causal-attention layers out of twelve.
Literature on sparse attention usually means sparse attention patterns inside a layer, so "a few
attention layers" is the less ambiguous phrase.

## 2. Pointers, copying, quoting and attributed generation

| Work | Authors, year | Venue / link | Shared with lexhybrid | Not shared |
|---|---|---|---|---|
| Pointer Networks | Vinyals, Fortunato, Jaitly, 2015 | [arXiv 1506.03134](https://arxiv.org/pdf/1506.03134) | outputs are positions in the input | combinatorial problems, not text or citations |
| Incorporating Copying Mechanism in Sequence-to-Sequence Learning (CopyNet) | Gu, Lu, Li, Li, 2016 | ACL 2016, [arXiv 1603.06393](https://arxiv.org/abs/1603.06393v1) | copying spans of the input | token-level soft copy, no citation |
| Get To The Point: Summarization with Pointer-Generator Networks | See, Liu, Manning, 2017 | ACL 2017, [P17-1099](https://arxiv.org/pdf/1704.04368) | copying from the source to reproduce facts accurately | token-level copy; nothing guarantees a verbatim span |
| Copy Is All You Need | Lan, Cai, Wang, Huang, Mao, 2023 | ICLR 2023, [arXiv 2307.06962](https://arxiv.org/abs/2307.06962) | generation by copying text segments from a collection | segments found by vector search; no citation labels or verifier |
| WebGPT | Nakano et al., 2021 | [arXiv 2112.09332](https://arxiv.org/abs/2112.09332) | answers backed by references collected from retrieved pages | free-text answers; human-feedback training |
| Teaching language models to support answers with verified quotes (GopherCite) | Menick et al., 2022 | [arXiv 2203.11147](https://arxiv.org/pdf/2203.11147) | special quote syntax; constrained sampling so quotes are verbatim spans of retrieved documents; abstention when unsure | 280B model generates the quote text under the constraint; RL from human preferences; English |
| Ground Every Sentence: Interleaved Reference-Claim Generation (ReClaim) † | Xia et al., 2024 | Findings of NAACL 2025, [arXiv 2407.01796](https://arxiv.org/html/2407.01796v2) | sentence-level citations; constrained decoding over a prefix tree of the retrieved passages' sentences so each reference is copied from the source | the model writes the reference text (not an index token); general-domain LLMs; no NLI gate described in the parts read |
| Enabling Large Language Models to Generate Text with Citations (ALCE) | Gao, Yen, Yu, Chen, 2023 | EMNLP 2023, [arXiv 2305.14627](https://arxiv.org/abs/2305.14627) | answers cite numbered retrieved passages; automatic citation-quality metrics | no constraint on decoding; benchmark rather than system |
| Attributed Question Answering | Bohnet et al., 2022 | [arXiv 2212.08037](https://arxiv.org/pdf/2212.08037) | attribution of answers to a corpus; AutoAIS, an NLI-based attribution metric | evaluation framework, not a constrained decoder |
| Effective LLM Adaptation for Improved Grounding and Citation Generation (AGREE) | Ye, Sun, Arik, Pfister, 2024 | NAACL 2024, [2024.naacl-long.346](https://aclanthology.org/2024.naacl-long.346) | fine-tuning so that claims carry citations to retrieved passages; NLI used to build training data | free-text citations, no constrained decoding |
| Self-RAG | Asai, Wu, Wang, Sil, Hajishirzi, 2024 | ICLR 2024, [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html) | special tokens added to the vocabulary to control retrieval and judge support | reflection tokens, not pointers into passages |
| Attribute First, then Generate | Slobodkin, Hirsch, Cattan, Schuster, Dagan, 2024 | ACL 2024, [2024.acl-long.182](https://aclanthology.org/2024.acl-long.182) | select source segments first; the selected segments are the attributions | multi-step LLM pipeline; summarization and long-form QA |
| Learning Fine-Grained Grounded Citations (FRONT) † | Huang et al., 2024 | Findings of ACL 2024 per arXiv, [arXiv 2408.04568](https://arxiv.org/abs/2408.04568) | ground the output in short supporting quotes from source documents before generating | fine-tuning method; no constrained decoding described |
| LongCite | Zhang, Bai et al., 2024 | Findings of ACL 2025, [arXiv 2409.02897](https://arxiv.org/abs/2409.02897) | sentence-level citations generated in one pass | long-context QA over one document; 8-9B models |
| Verifiable by Design (Quote-Tuning) | Zhang, Marone, Li, Van Durme, Khashabi, 2024 | NAACL 2025, [arXiv 2404.03862](https://arxiv.org/abs/2404.03862v2) | verbatim quoting from trusted text | quotes from pretraining memory, not retrieval |
| GenProve † | Wei et al., 2026 | [arXiv 2601.04932](https://arxiv.org/abs/2601.04932) | sentence-level provenance with distinct relation types (Quotation, Compression, Inference), close to lexhybrid's quote/cite distinction | no constrained decoding or verifier described in the abstract; SFT + GRPO |
| ACL-Verbatim (VerbatimRAG applied to ACL papers) † | Recski, Tóth, Verdha, Boros, Kovács, 2026 | [arXiv 2605.21102](https://arxiv.org/abs/2605.21102) | answers are exact spans of retrieved documents | extractive span selection (token classifier or LLM extractor) |
| Attribution, Citation, and Quotation: A Survey † | Schreieder, Schopf, Färber, 2025 | ACL 2026 per arXiv, [arXiv 2508.15396](https://arxiv.org/abs/2508.15396) | taxonomy covering in-generation citation and quotation | survey |

**Finding.** Every ingredient of lexhybrid's answer format is published: pointing into the input
(Pointer Networks, CopyNet), verbatim quoting enforced at decoding time (GopherCite, ReClaim),
citations to numbered retrieved passages (ALCE) and sentence-level citation (LongCite, ReClaim,
GenProve). The specific realization, dedicated passage and sentence index tokens under a finite-state
logit mask, with a renderer that substitutes the source sentence and its statutory label, was not
found as such. It is closest to ReClaim and GopherCite and must be presented as a variant of
constrained verbatim quoting, not as a new idea.

## 3. Answer verification for RAG, general and legal

| Work | Authors, year | Venue / link | Shared with lexhybrid | Not shared |
|---|---|---|---|---|
| Measuring Attribution in Natural Language Generation Models (AIS) | Rashkin et al., 2023 | Computational Linguistics 49(4), [2023.cl-4.2](https://aclanthology.org/2023.cl-4.2.pdf), [arXiv 2112.12870](https://arxiv.org/abs/2112.12870v2) | defines when a statement is attributable to an identified source | human annotation framework |
| TRUE: Re-evaluating Factual Consistency Evaluation | Honovich et al., 2022 | NAACL 2022, [2022.naacl-main.287](https://aclanthology.org/2022.naacl-main.287) | large NLI models are strong factual-consistency checkers | evaluation study |
| AutoAIS (in Attributed QA) | Bohnet et al., 2022 | [arXiv 2212.08037](https://arxiv.org/pdf/2212.08037) | NLI-based attribution score with a 0.5 threshold | metric for evaluation and reranking, not an answer gate |
| RARR | Gao, Dai et al., 2023 | ACL 2023, [arXiv 2210.08726](https://arxiv.org/abs/2210.08726v3) | finds evidence and edits unsupported content | revises with an LLM instead of dropping sentences or refusing |
| Automatic Evaluation of Attribution by LLMs (AttrScore) | Yue, Wang, Chen, Zhang, Su, Sun, 2023 | Findings of EMNLP 2023, [arXiv 2305.06311](https://arxiv.org/abs/2305.06311v2) | checks whether a cited reference supports a statement | LLM or fine-tuned evaluators; evaluation |
| MiniCheck | Tang, Laban, Durrett, 2024 | EMNLP 2024, [2024.emnlp-main.499](https://aclanthology.org/2024.emnlp-main.499) | small checker of LLM output against grounding documents | Flan-T5 checker trained on synthetic data; English |
| mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 | Laurer (model card) | [Hugging Face](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) | the NLI model lexhybrid's verifier uses | the card notes machine-translated training data |
| Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools | Magesh, Surani, Dahl, Suzgun, Manning, Ho, 2025 | J. Empirical Legal Studies 22(2), [Stanford page](https://law.stanford.edu/publications/hallucination-free-assessing-the-reliability-of-leading-ai-legal-research-tools), [arXiv 2405.20362](https://arxiv.org/abs/2405.20362v1) | legal RAG; shows that "hallucination-free" vendor claims were overstated (17% to 33% hallucination in the preprint) | evaluation of US commercial tools, not a method |
| Large Legal Fictions | Dahl, Magesh, Suzgun, Ho, 2024 | J. Legal Analysis 16(1), [arXiv 2401.01301](https://arxiv.org/pdf/2401.01301) | legal hallucination typology | LLMs without retrieval; US case law |
| HalluGraph † | Noël, Seidou, Capo-Chichi, Amari, 2025 | under review, [arXiv 2512.01659](https://arxiv.org/abs/2512.01659) | hallucination detection for legal RAG; entity grounding acts as a citation check; NLI used as a baseline | knowledge-graph alignment scoring; no jurisdiction or language stated |
| CitaLaw | Zhang, Yu, Dai, Xu, 2024 | Findings of ACL 2025, [arXiv 2412.14556](https://arxiv.org/pdf/2412.14556) | legal answers with law-article and precedent citations | benchmark; Chinese legal system |
| LegalBench-RAG | Pipitone, Houir Alami, 2024 | [arXiv 2408.10343](https://arxiv.org/html/2408.10343v1) | legal retrieval tied to exact source fragments | evaluates retrieval only; English contracts and privacy texts |
| How Much Do Legal RAG Systems Still Hallucinate? † | Das, Abualhaija, Bianculli, 2026 | [arXiv 2608.14210](https://arxiv.org/abs/2608.14210) | claim-level evaluation of legal RAG systems | EU (English) and French civil law; evaluation |
| Fine-grained Claim-level RAG Benchmark for Law (ClaimRAG-LAW) † | Das, Abualhaija, Bianculli, 2026 | [arXiv 2605.21071](https://arxiv.org/abs/2605.21071) | claim-level legal RAG evaluation | French and English; benchmark |

**Finding.** NLI-based attribution checking (AIS, TRUE, AutoAIS), post-hoc revision (RARR) and
abstention (GopherCite) are standard. Legal RAG hallucination is measured (Magesh et al., Das et al.)
and detected (HalluGraph). No legal RAG system was found that gates every answer by pointer
resolution, a verbatim quote check, per-claim NLI and a person-name refusal together, and none for
German law; each individual check is established practice.

## 4. German and DACH legal NLP

| Work | Authors, year | Venue / link | Shared with lexhybrid | Not shared |
|---|---|---|---|---|
| Answering legal questions from laymen in German civil law system (GerLayQA) | Büttner, Habernal, 2024 | EACL 2024, [2024.eacl-long.122](https://aclanthology.org/2024.eacl-long.122) | German lay questions with lawyer answers grounded in statute paragraphs; retrieval and generation baselines | dataset and baselines; GPT-3.5 generation |
| GerDaLIR: A German Dataset for Legal Information Retrieval | Wrzalik, Krechel, 2021 | NLLP 2021, [2021.nllp-1.13](https://aclanthology.org/2021.nllp-1.13), [GitHub](https://github.com/lavis-nlp/GerDaLIR) | German case-law retrieval from Open Legal Data; BM25 and neural reranker baselines | retrieval benchmark only |
| Towards Intelligent Legal Advisors for Document Retrieval and Question-Answering in German Legal Documents | Hoppe, Pelkmann, Migenda, Hötte, Schenck, 2021 | IEEE AIKE 2021, [dblp](https://dblp.dagstuhl.de/rec/conf/aike/HoppePMHS21.html) | German legal QA over court decisions; lexhybrid's LegalQuAD loader reads the authors' AIKE 2021 appendix release | not confirmed from the paper that it introduces the name LegalQuAD |
| GerLeRB † | creator listed as anonymous, 2025 | [Zenodo record 15745124](https://zenodo.org/records/15745124), CC BY 4.0 | German statute retrieval benchmark (TREC format) | no associated paper found |
| BenGER † | Nagl et al., 2026 | EMNLP 2026 per arXiv, [arXiv 2605.28183](https://arxiv.org/abs/2605.28183) | German law; LLM evaluation | subsumption-style reasoning tasks, no retrieval citations |
| Chunking German Legal Code † | Prior, Milanova, Schultz, 2026 | ASAIL workshop at ICAIL 2026, [arXiv 2605.19806](https://arxiv.org/abs/2605.19806) | RAG over German statutes; structure-aligned chunks (sections, subsections) retrieve best | chunking study only |
| Bundesrecht † | Darji, Heckelmann, Kratsch, de Melo, 2026 | [arXiv 2605.31338](https://arxiv.org/abs/2605.31338) | German federal statutes with hierarchy down to subclauses; normalized citation strings | reference parsing library, no QA model |
| Domain-Adaptation through Synthetic Data: Fine-Tuning LLMs for German Law † | Bashir et al., 2026 | [arXiv 2601.14160](https://arxiv.org/abs/2601.14160) | German legal QA from statutes | parameter-efficient fine-tuning of existing LLMs; no citation pointers or verifier described |
| SteuerLLM † | Wind et al., 2026 | Scientific Reports 16:21640, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13354760/) | German legal-domain LLM | tax law; 28B, block expansion of Mistral Small; no retrieval at inference |
| A Dataset of German Legal Documents for Named Entity Recognition | Leitner, Rehm, Moreno-Schneider, 2020 | LREC 2020, [2020.lrec-1.551](https://aclanthology.org/2020.lrec-1.551) | the dataset behind the German legal NER model (`flair/ner-german-legal`) lexhybrid's scrub uses | NER only |
| Towards an Open Platform for Legal Information (Open Legal Data) | Ostendorff, Blume, Ostendorff, 2020 | JCDL 2020, [arXiv 2005.13342](https://arxiv.org/abs/2005.13342v1) | source platform of the `oldp` corpus | data platform |
| MultiLegalPile | Niklaus et al., 2024 | ACL 2024, [2024.acl-long.805](https://preview.aclanthology.org/setup/2024.acl-long.805) | multilingual legal corpus used for the research arm; legal encoder models | encoders; no generation |
| LEXTREME | Niklaus et al., 2023 | Findings of EMNLP 2023, [arXiv 2301.13126](https://arxiv.org/abs/2301.13126) | multilingual legal benchmark | classification-style tasks |
| Noxtua (Xayn with CMS; Beck-Noxtua) | vendor | [press release](https://www.noxtua.com/de/news/press-releases/launch-of-the-european-legal-ai-noxtua), [FAQ](https://www.noxtua.com/faq) | German legal AI with a proprietary legal language model | no architecture, size or citation mechanism published on the pages found |

**Finding.** German legal QA and retrieval benchmarks exist and must be cited where they are used.
The German legal LLMs found are adapted from general Transformer LLMs (SteuerLLM, Bashir et al.); none
is trained from scratch, none uses a recurrent or SSM backbone, and none answers through constrained
pointer citations with a verifier. No commercial German legal product was found that publishes its
model or citation technique. A web search for juris, Wolters Kluwer and Beck legal AI technical papers
returned none.

## 5. Small domain-specific legal models, training from scratch, distillation

| Work | Authors, year | Venue / link | Shared with lexhybrid | Not shared |
|---|---|---|---|---|
| LEGAL-BERT | Chalkidis et al., 2020 | Findings of EMNLP 2020, [2020.findings-emnlp.261](https://aclanthology.org/2020.findings-emnlp.261) | legal model pretrained from scratch (LEGAL-BERT-SC) | encoder; English; no distillation |
| Pile of Law | Henderson et al., 2022 | NeurIPS 2022 Datasets and Benchmarks, [arXiv 2207.00220](https://arxiv.org/abs/2207.00220v1) | legal pretraining corpus with filtering for private content | English |
| SaulLM-7B | Colombo et al., 2024 | [arXiv 2403.03883](https://arxiv.org/pdf/2403.03883) | legal LM pretraining with general replay data | continued pretraining of Mistral 7B; English-dominant |
| Ayn: A Tiny yet Competitive Indian Legal Language Model Pretrained from Scratch † | Niyogi, Gaussier, Bhattacharya, 2024 | LREC 2026 per arXiv, [arXiv 2403.13681](https://arxiv.org/abs/2403.13681) | small (88M) decoder legal LM pretrained from scratch with a domain tokenizer | Indian law; no distillation mentioned |
| Gemma 2 | Gemma Team, 2024 | [arXiv 2408.00118](https://arxiv.org/abs/2408.00118v3) | 2B and 9B models pretrained with knowledge distillation instead of next-token prediction | general domain; Transformer |
| Pre-training Distillation for LLMs: A Design Space Exploration | Peng et al., 2024 | ACL 2025, [2025.acl-long.181](https://aclanthology.org/2025.acl-long.181) | logit distillation during pretraining, online and offline logits | general domain |
| MiniPLM | Gu et al., 2024 | ICLR 2025, [arXiv 2410.17215](https://arxiv.org/abs/2410.17215v1) | students of 200M to 1.2B pretrained from scratch with a teacher | distills through data selection, not logits |
| Qwen3 Technical Report | Qwen Team, 2025 | [arXiv 2505.09388](https://arxiv.org/abs/2505.09388) | tokenizer and distillation teacher | general domain |
| Scaling Legal AI: Benchmarking Mamba and Transformers † | Maurya, 2025 | [arXiv 2509.00141](https://arxiv.org/abs/2509.00141) | Mamba-family models on legal text | classification and retrieval only; no generation |

Distillation into hybrid students is covered in section 1 (Mamba in the Llama; Hauzenberger et al.
2026); both derive the student from the pretrained teacher (reused attention weights,
linearization), whereas lexhybrid's student is randomly initialized.

**Finding.** Small legal LMs trained from scratch exist (Ayn, LEGAL-BERT), and logit distillation
during pretraining is published (Gemma 2, Peng et al.). No legal LM was found that is trained from
scratch with logit distillation, and none for German.

## Verdict

**Already published, so must be cited, not claimed:**

- Interleaving SSM layers with a few attention layers: Jamba, Zamba, Samba, the Mamba-2-Hybrid study,
  Nemotron-H, Hymba, Falcon-H1, Bae et al.
- Interleaving mLSTM/xLSTM layers with attention: SWAX, Hauzenberger et al. 2026, Hartl et al. 2026.
- Distilling a Transformer teacher into a hybrid student: Mamba in the Llama, Hauzenberger et al. 2026.
- Constrained decoding that forces verbatim quotes from retrieved text: GopherCite, ReClaim.
- Citations by passage number and at sentence level: ALCE, LongCite, ReClaim, GenProve, FRONT.
- NLI-based attribution checking, revision and abstention: AIS, TRUE, AutoAIS, RARR, GopherCite.
- Legal RAG hallucination: Magesh et al., Dahl et al.
- Small legal LMs from scratch and pretraining distillation: Ayn, LEGAL-BERT, Gemma 2, Peng et al.
- German legal data: GerLayQA, GerDaLIR, the LegalQuAD release, GerLeRB, the LER NER dataset, Open
  Legal Data.

**Not found:**

- A language model combining Mamba (any version), mLSTM and softmax-attention layers in one stack.
  The closest is Hartl et al. 2026, which pairs each recurrent operator with attention in a separate
  model.
- The full combination: a three-way Mamba-3 + mLSTM + attention model, trained from scratch with
  logit distillation, answering German legal questions through pointer tokens under a constrained
  decoder and an NLI-based verifier gate.

**Licensed wording:**

- Allowed: "To our knowledge, no published language model combines Mamba-3, mLSTM and softmax-attention
  layers in one stack (literature search of 2026-11-03)." Name Hartl et al. 2026 next to it as the
  closest work.
- Allowed: "We are not aware of prior work that combines a Mamba-3/mLSTM/attention hybrid, pointer-token
  citations under constrained decoding and an NLI verifier for German law." "First" may be used only
  with "to our knowledge", only for this full combination or the three-way layer mix, and only
  together with the date of the search.
- Not allowed: "first" for any single part: hybrid SSM-attention models, hybrid xLSTM-attention models,
  constrained or verbatim quoting, pointer or sentence-level citations, NLI verification, legal RAG,
  German legal AI, or a small legal model trained from scratch. Each has prior work above.
- Not allowed: "hallucination-free" or "guaranteed correct". Constrained decoding guarantees that
  a pointer resolves, and the quote check guarantees that a quote is verbatim; whether a claim is
  entailed is decided by an NLI model at a 0.5 threshold, which can be wrong. Magesh et al. document
  why the term "hallucination-free" misleads in legal RAG.
- A novelty statement about the layer mix says nothing about its merit. `analysis/screen_results.md`
  removes attention but includes no arm without mLSTM or without Mamba-3, so no claim that the
  three-way mix beats a two-way hybrid is supported.

**Works the README must cite** (it currently acknowledges Mamba-3, xLSTM, TFLA and Qwen3):

1. Mamba-3 (Lahoti et al. 2026); Mamba-2/SSD (Dao, Gu 2024) for the SSD scan.
2. xLSTM (Beck et al. 2024); TFLA (Beck et al. 2025).
3. Hybrid precedents: Jamba (Lieber et al. 2024), Samba (Ren et al. 2024), the Mamba-2-Hybrid study
   (Waleffe et al. 2024), Hymba (Dong et al. 2024); xLSTM + attention: Hartl et al. 2026 and
   Hauzenberger et al. 2026.
4. Distillation: Gemma 2 (2024) or Peng et al. (2025) for pretraining distillation; Qwen3 (2025) as
   teacher and tokenizer.
5. Quoting and citation: GopherCite (Menick et al. 2022), ReClaim (Xia et al. 2024), ALCE (Gao et al.
   2023).
6. Verification: AIS (Rashkin et al. 2023), Attributed QA / AutoAIS (Bohnet et al. 2022), RARR (Gao
   et al. 2023), and the mDeBERTa NLI model card.
7. Legal RAG reliability: Magesh et al. (2025).
8. Retrieval components: BGE M3 (M3-Embedding, Chen et al. 2024,
   [arXiv 2402.03216](https://arxiv.org/abs/2402.03216v1)), the `BAAI/bge-reranker-v2-m3` model, and
   reciprocal rank fusion (Cormack, Clarke, Büttcher, SIGIR 2009,
   [record](https://ir.webis.de/anthology/2009.sigirconf_conference-2009.146/)).
9. Evaluation and scrub data: GerLayQA, GerDaLIR, the LegalQuAD release (Hoppe et al. 2021), GerLeRB
   (Zenodo record), the LER dataset (Leitner et al. 2020).

## Not verified

- Mamba-3: only the abstract page was read; whether the paper reports hybrids with attention is open.
- Hartl et al. 2026: the last part of the HTML text (about 43k characters) was not read.
- ReClaim: the languages of its evaluation sets were not checked.
- xLSTM 7B: whether it contains attention layers was not checked beyond the abstract text in search
  results.
- ALCE: which NLI model its citation metrics use was not checked.
- GerLeRB: the Zenodo record lists its creator as anonymous and names no paper; the attribution
  "Weber, GI 2025" in `docs/CORPUS_LICENCE_REGISTER.md` could not be confirmed.
- LegalQuAD: the dblp record of Hoppe et al. (AIKE 2021) was found, but the paper itself was not
  opened, so it is unconfirmed that it introduces the name.
- VerbatimRAG: the original VerbatimRAG paper was not found; only the 2026 ACL-Verbatim paper was read.
- LEXam (law exams in English and German) appeared only on secondary pages and is not listed above.
- Commercial products: no technical documentation was found for Noxtua, Beck, juris or Wolters Kluwer
  legal AI. That shows only that none was found, not that none exists.
