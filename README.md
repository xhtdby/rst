# rst_trap_finder

Analyze a word-association graph to locate **death words** – prompts that funnel an opponent's next reply to start with letters in the trap set R, S or T.

## Scores
- **S1(u)** – probability next reply starts with R/S/T.
- **Escape Hardness** – fraction of strong edges leading to R/S/T.
- **Biased PageRank** – PageRank with edges to R/S/T boosted.
- **KStepRST** – probability of reaching R/S/T within *k* steps.
- **MinimaxTopM** – worst-case/soft score if opponent chooses among top-*m* edges.
- **Composite** – weighted sum of the above.

## Quickstart
```bash
pip install -e .
python -m rst_trap_finder.cli rank --csv data/edges.sample.csv --top 15
python -m rst_trap_finder.cli next --word color --csv data/edges.sample.csv
```

CSV schema: `src,dst,weight` with lower‑case tokens and positive weights.
