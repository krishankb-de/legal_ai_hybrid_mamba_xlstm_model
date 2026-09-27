# Document schema

Every collector emits `lexhybrid.data.schema.Document` records, one JSON object per line. Every
later stage reads them: the manifests, the LER scrub, deduplication, packing, the retrieval index,
the renderer and the verifier. The schema is plan box P3-A.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `id` | str | `<source>:<native id>`, stable across re-collection, for example `gii:bgb:573` or `rii:KORE312022021` |
| `source` | str | the collector that produced it: `gii`, `rii`, `oldp`, `eurlex`, `ris`, `fedlex`, `bger`, `dip`, `fineweb2_de`, `multilegalpile` |
| `jurisdiction` | str | `DE`, `AT`, `CH` or `EU`; general-web German text is `DE` |
| `doc_type` | str | `statute`, `regulation`, `decision`, `parliament` or `general` |
| `licence` | str | licence id from `Docs/CORPUS_LICENCE_REGISTER.md`; `unknown` is allowed in the record but the corpus build refuses it |
| `commercial_safe` | bool | may enter a shippable training arm |
| `research_only` | bool | belongs to the research-exception arm (decision 2); never `commercial_safe` |
| `text` | str | the full text, exactly as published after whitespace normalisation |
| `citation_id` | str | how the document is cited; grammar below |
| `url` | str | where it was retrieved |
| `valid_from`, `valid_to` | ISO date or null | the version's validity window: a statute's `Stand` or entry-into-force date; a decision's date in `valid_from` |
| `retrieved_at` | ISO timestamp | UTC time of retrieval |
| `sha256` | str | SHA-256 of the UTF-8 `text`; filled automatically and checked on load |
| `sections` | list of `Section` | the quotable units, in document order |

`Section` holds `label` (the citable label inside the document, such as `§ 573 Abs. 2 Nr. 1`,
`Art. 97 Abs. 1`, `Rn. 14`, `E. 3.2` or `Tenor`), `absatz`, `satz_idx`, `randnummer`, and the section
`text`. The integer fields are null when they do not apply.

A record that breaks these rules raises `ValueError` on construction and on load:
- `id` must start with `source:`.
- The jurisdiction, document type and dates must be valid.
- A `research_only` record must not be `commercial_safe`.
- `sha256` must match `text`.

## Citation ids

Three grammars, parsed by `lexhybrid.data.schema.parse_citation`.

**Statutes and regulations:** `<code> §<n>` or `<code> Art. <n>`, where `<code>` is the act's
abbreviation (or `SR <number>`) and may be followed by `SchlT`; then, optionally,
`Abs. <k>`, then `Satz <s>` or `S. <s>`, then `Nr. <m>`, then `lit. <x>`. For example:
- `BGB §573 Abs. 2 Nr. 1`: German Civil Code
- `ABGB §1295`: Austrian General Civil Code
- `OR Art. 97`: Swiss Code of Obligations
- `DSGVO Art. 6 Abs. 1 lit. f`: GDPR, in its German short name
- `ZGB SchlT Art. 1`: an article of the Swiss Civil Code's final title (Schlusstitel)
- `SR 221.214.111 Art. 3`: a Swiss act without an abbreviation, cited by its SR number

Numbers may carry a letter suffix: `BGB §573a`, `ZGB Art. 334bis`, `OR Art. 40a Abs. 2bis`.

The renderer (P7) prints these labels next to every quoted sentence.

**Decisions:** `<court> <docket>`, optionally followed by `Rn. <n>` or `Rz. <n>` for a margin number, or by `E. <n.n>` for a Swiss Erwägung. For example:
- `BGH VIII ZR 12/20 Rn. 14`
- `OGH 1 Ob 123/20x`
- `BGer 4A_123/2020 E. 3.2`

The court is one of `BGH`, `BVerfG`, `BVerwG`, `BFH`, `BAG`, `BSG`, `BPatG`, `OGH`, `VfGH`,
`VwGH`, `BGer`, `BVGer`, `EuGH`, `EuG`, or another abbreviation ending in `G` or `GH`, such as `OLG`, `LG`, `VG`, `VGH` or `VerfGH`. A lower court's seat is written into the docket, for example `VG Bremen 2 K 1343/24` or `OLG Nürnberg 8 W 1488/26 Ver`.

The `citation_id` of a whole document cites the document, for example `BGB §573`. Section labels
carry the finer grain, for example `Abs. 2 Nr. 1`.

**Parliamentary papers:** `BT-Drs.`, `BR-Drs.`, `BT-PlPr.` or `BR-PlPr.`, then the paper's number,
optionally followed by `S. <page>`. For example `BT-Drs. 21/8109` or `BR-Drs. 506/26`; the DIP terms
of use prescribe this form when a paper is quoted.
