# PersonaMem-v2 first-pass development results

**Status note:** these are development diagnostics over the legacy
`benchmark_personamem_v0` slice, not benchmark results. A source-support
audit later found 13 of the 30 labeled personal-memory claims unsupported by
the raw indexed conversation, so this table cannot support a canonical
PARMBench claim. See `docs/benchmark-construction.md` for the canonical
contract.

This is the frozen first pass over 30 persona-disjoint scenarios and 90 cases. P is positive accuracy, C is cue-ablated control accuracy, and Z is the memory-included ceiling.

| Condition | P | C | Z | Admission precision | Admission recall | Spurious | Poison | Stale or contradiction | Privacy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No memory | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Enhanced input RAG | 16.7% | 86.7% | 100.0% | 0.7% | 6.7% | 99.3% | 0.0% | 0.0% | 0.0% |
| Hybrid output RAG | 13.3% | 76.7% | 100.0% | 1.0% | 16.7% | 99.0% | 0.0% | 0.0% | 0.0% |
| Exact-entity output RAG | 6.7% | 76.7% | 100.0% | 1.2% | 33.3% | 98.8% | 0.0% | 0.0% | 0.0% |
| Prompted memory tool | 3.3% | 100.0% | 100.0% | 20.0% | 3.3% | 80.0% | 0.0% | 0.0% | 0.0% |
| PARM | 16.7% | 93.3% | 100.0% | 50.0% | 13.3% | 50.0% | 0.0% | 0.0% | 0.0% |

## First-pass readout

- PARM and enhanced input RAG tie for the highest positive accuracy at 16.7%. PARM keeps 93.3% of controls stable versus 86.7% for input RAG.
- PARM admits little evidence, but its 50.0% precision is substantially cleaner than every fixed top-k condition. Its 13.3% recall remains too low for strong coverage.
- Exact-entity output RAG reaches 33.3% recall, but 98.8% of its admissions are non-gold and it keeps only 76.7% of controls stable.
- Every condition reaches the 100.0% explicit-memory ceiling. Together with the no-memory 0/30 positive, 30/30 control, and 30/30 ceiling gate, this makes retrieval and admission the main first-pass bottleneck rather than basic task usability.

## Interpretation boundaries

- This development set uses ordinary, current, self-attributed preferences. Poison, stale-source, and privacy rates are regression checks here, not a safety stress-test result.
- The 9k-token recommendation feeds are deterministic fixtures built for exact cue/control symmetry. They do not establish natural end-to-end validity.
- No LLM judge was used for this first pass because all 30 triplets passed the no-memory gate without manual repair. If later batches require accumulating repairs or topic-specific templates, construction should move to a versioned judge rubric calibrated against sampled human review.
- The fixtures and fairness artifacts were committed before retrieval failures were inspected. The frozen Amara benchmark was not modified.

## Per-persona corpus results

Each cell is P/C/Z/R, where R is gold-memory admission recall.

| Corpus | Scenario | No memory | Enhanced input RAG | Hybrid output RAG | Exact-entity output RAG | Prompted memory tool | PARM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `personamem-v2/train/persona-121` | family-recipes | 0/1/1/0 | 0/0/1/0 | 0/0/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-171` | multicultural-food-festival | 0/1/1/0 | 0/0/1/1 | 0/1/1/0 | 0/0/1/1 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-188` | mosque-study-circle | 0/1/1/0 | 1/1/1/0 | 0/0/1/0 | 0/1/1/0 | 0/1/1/0 | 1/1/1/0 |
| `personamem-v2/train/persona-192` | african-proverbs | 0/1/1/0 | 0/0/1/0 | 0/0/1/0 | 1/1/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-214` | sustainable-design | 0/1/1/0 | 0/1/1/0 | 1/1/1/1 | 0/1/1/1 | 0/1/1/0 | 1/0/1/1 |
| `personamem-v2/train/persona-38` | seasonal-nature | 0/1/1/0 | 0/1/1/1 | 0/1/1/0 | 0/1/1/1 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-393` | off-grid-hot-springs | 0/1/1/0 | 0/1/1/0 | 1/1/1/1 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-413` | authentic-sushi | 0/1/1/0 | 1/1/1/0 | 0/0/1/0 | 1/0/1/1 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-423` | family-photo-display | 0/1/1/0 | 0/1/1/0 | 0/1/1/1 | 0/0/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-461` | meditation-app | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/1 | 0/1/1/0 | 1/1/1/1 |
| `personamem-v2/train/persona-469` | three-day-gym | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/0/1/0 |
| `personamem-v2/train/persona-490` | vegan-leather-bag | 0/1/1/0 | 0/1/1/0 | 0/1/1/1 | 0/0/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-552` | restorative-yoga | 0/1/1/0 | 1/1/1/0 | 1/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-553` | wine-cellar | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-629` | group-fitness | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/1 | 0/1/1/0 | 1/1/1/1 |
| `personamem-v2/train/persona-642` | craft-beer-tasting | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-726` | monochrome-photography | 0/1/1/0 | 1/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-756` | puerto-rican-cooking | 0/1/1/0 | 0/1/1/0 | 1/0/1/0 | 0/0/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-819` | screen-free-weekend | 0/1/1/0 | 1/1/1/0 | 0/1/1/0 | 0/0/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-854` | morning-run | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-855` | challenging-puzzle | 0/1/1/0 | 0/1/1/0 | 0/0/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-861` | environmental-charity | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 1/1/1/1 | 0/1/1/0 |
| `personamem-v2/train/persona-869` | plain-spice-jars | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-877` | experimental-instruments | 0/1/1/0 | 0/1/1/0 | 0/0/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-887` | historical-drama | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 1/1/1/1 |
| `personamem-v2/train/persona-945` | european-current-affairs | 0/1/1/0 | 0/1/1/0 | 0/1/1/1 | 0/1/1/1 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-956` | morning-weightlifting | 0/1/1/0 | 0/0/1/0 | 0/1/1/0 | 0/0/1/0 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-98` | leisure-cycling | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/1 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-981` | bookstore-browsing | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/1 | 0/1/1/0 | 0/1/1/0 |
| `personamem-v2/train/persona-983` | assertive-business | 0/1/1/0 | 0/1/1/0 | 0/1/1/0 | 0/1/1/1 | 0/1/1/0 | 0/1/1/0 |

The JSON comparison beside the predictions contains the full per-corpus metric objects, exact configuration fields, and content hashes for every result artifact.
