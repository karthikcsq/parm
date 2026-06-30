# Amara Life source fixture

This directory vendors the compact `amara-life-v1` source fixture from
`garrytan/gbrain-evals` for reproducible PARM experiments. The original corpus
manifest identifies the fixture as MIT licensed and generated with seed 42.

The raw source is intentionally tracked. Generated Markdown pages, GBrain
runtime code, PGLite data, embeddings, model weights, and caches remain under
the ignored `.gbrain-local/` directory.

`src/parm_bench/amara.py` converts this source into provenance-preserving
Markdown. Perturbation labels such as `poison`, `contradiction`, and
`stale-fact` are retained in frontmatter and must not silently become
authoritative memory.
