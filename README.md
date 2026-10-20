# Benefit and Cost of Random-Token Replacement under Repeated Data

Code for the paper **Benefit and Cost of Random-Token Replacement under Repeated Data: A Known-Entropy Decomposition and Evidence from Ten Real Corpora** (Wonsup Shin and Jingu Kim, Flitto).

All models are trained from scratch (134M / 336M decoder-only Transformers) under one fixed recipe, on a single L4 or L40S GPU in about three hours per run. Every corpus is public, and the model, training loop, controlled Markov source, and all measurement scripts are included here.

## What reproduces what

| Paper element | Script(s) |
|---|---|
| Model (336M / 134M decoder-only Transformer) | `code/model.py` |
| Training, `--aug none` vs `--aug replace`, all schedules | `code/train.py` |
| Controlled order-1 Markov source with known entropy rate | `code/markov_source.py` |
| 64k multilingual tokenizer | `code/train_tokenizer.py` |
| Corpus builders (Sec. 3, App. A) | `code/build_hf_corpus.py`, `build_yodas_transcript.py`, `build_code_bin.py`, `build_sql_bin.py`, `build_loghub_bin.py` |
| Loss decomposition H + APPROX + EST + MEM (Sec. 5, App. B.1) | `code/decompose.py` |
| Raw-text gzip of a corpus, survey over 64 further corpora (Sec. 7, Figure 5, App. B.2) | `code/redundancy_survey.py`, `survey_manifest.json` |
| Bigram MI of the trained corpora on their token `.bin` (Table 5) | `code/redundancy_stats.py` |
| Higher-order conditional entropy (App. B.3) | `code/measure_cond_entropy.py` |
| Memorization and shrinkage probes on the Markov source (App. C.1) | `code/probe_memorization.py` |
| Shrinkage weight of replacement on each corpus (App. C.3) | `code/shrinkage_weight.py` |
| Early-stopping simulation on validation curves (App. C.4) | `code/early_stopping.py` |

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10, CUDA 12.x, one L4 or L40S GPU. No multi-GPU or multi-node setup is required.

## 1. Tokenizer

One shared 64k SentencePiece tokenizer over Korean, English, and Japanese:

```bash
python code/train_tokenizer.py --input corpus.txt --vocab 64000 --model-prefix tok_kej_64k
```

## 2. Data

Each corpus is tokenized to `train.bin` / `val.bin` (uint16) with a fixed training-token budget (App. A).

| Corpus | Public source | Train tokens | Builder |
|---|---|---|---|
| Korean web | `HuggingFaceFW/fineweb-2`, `kor_Hang` | 7.4M | `build_hf_corpus.py` |
| Japanese web | `HuggingFaceFW/fineweb-2`, `jpn_Jpan` | 7.4M | `build_hf_corpus.py` |
| Korean speech transcripts | YODAS (`espnet/yodas`) | 7.4M | `build_yodas_transcript.py` |
| code | The Stack | 7.4M | `build_code_bin.py` |
| chess | `adamkarvonen/chess_games`, field `transcript` | 5.2M | `build_hf_corpus.py` |
| SQL-comb / SQL-ctx | `b-mc2/sql-create-context` | 5.0M / 2.3M | `build_sql_bin.py --mode combined / context` |
| census corpus (JSON) | `scikit-learn/adult-census-income`, rows serialized to JSON | 5.5M | `build_hf_corpus.py --json-dump` |
| road-safety corpus (JSON) | `inria-soda/tabular-benchmark`, `clf_cat_road-safety`, rows serialized to JSON | 7.4M | `build_hf_corpus.py --json-dump` |
| system-log corpus | `bolu61/loghub_2` | 7.4M | `build_loghub_bin.py` |
| Markov source | generated, constrained 2.0M / abundant 140M | | `markov_source.py` |

Examples:

```bash
T=tok_kej_64k.model
python code/build_hf_corpus.py --tokenizer $T --dataset HuggingFaceFW/fineweb-2 --config kor_Hang \
    --target-tokens 7383561 --val-tokens 500000 --out-dir data_bin_koweb
python code/build_hf_corpus.py --tokenizer $T --dataset adamkarvonen/chess_games --field transcript \
    --target-tokens 5200000 --val-tokens 300000 --out-dir data_bin_chess
python code/build_hf_corpus.py --tokenizer $T --dataset scikit-learn/adult-census-income --json-dump \
    --target-tokens 5500000 --val-tokens 300000 --out-dir data_bin_census
python code/build_hf_corpus.py --tokenizer $T --dataset inria-soda/tabular-benchmark --config clf_cat_road-safety --json-dump \
    --target-tokens 7400000 --val-tokens 300000 --out-dir data_bin_roadsafety
python code/build_sql_bin.py --tokenizer $T --mode context --target-tokens 2300000 --val-tokens 180000 --out-dir data_bin_sqlctx

# Markov source: order-1 chain over 1,024 states, temperature sets the entropy rate
python code/markov_source.py --temperature 0.5 --n-tokens 2000000 --val-tokens 200000 --out-dir data_bin_mk_t05          # data-constrained
python code/markov_source.py --temperature 0.5 --n-tokens 140000000 --out-dir data_bin_mk_t05_abund                      # abundant-data reference
```

The paper uses temperatures 0.2, 0.3, 0.5, 0.8, and 2.0 (`t02` to `t20`), with seed 0 for the source.

## 3. Training

The main recipe: constant learning rate `3e-4`, 300 warmup steps, AdamW (betas 0.9/0.95), weight decay 0.1, gradient clipping 1.0, batch 8, sequence length 2048, 10,000 steps, bfloat16. On the Markov source: `--vocab-size 1025 --max-iters 8000 --eval-interval 250`. Random-token replacement (`--aug replace`) replaces each input token with a uniform random vocabulary token at rate `--aug-p 0.15`, leaving the prediction targets intact.

```bash
ARCH="--n-layer 24 --n-embd 1024 --n-head 16 --n-kv-head 4 --ffn-hidden 2816"   # 336M; 134M: 14 / 768 / 12 / 2 / 2048
COMMON="--vocab-size 64001 --schedule constant --lr 3e-4 --min-lr 3e-4 --warmup 300 \
        --max-iters 10000 --eval-interval 300 --batch 8 --block-size 2048 --grad-checkpoint --weight-decay 0.1"

python code/train.py $ARCH $COMMON --data-dir data_bin_koweb --out-dir runs/koweb_none    --aug none    --seed 1337
python code/train.py $ARCH $COMMON --data-dir data_bin_koweb --out-dir runs/koweb_replace --aug replace --aug-p 0.15 --seed 1337
```

Seeds are 1337, 1338, and 1339; the paper marks the few comparisons that use two seeds. Each run writes its validation loss at every evaluation to `metrics.jsonl` and its latest checkpoint to `ckpt.pt`. The minimum loss and the final loss are the lowest and the last values in `metrics.jsonl`.

Variants used in Sec. 6 and App. C:

| Experiment | Change to the main recipe |
|---|---|
| Learning-rate decay over the full budget (Table 2) | `--schedule cosine --min-lr 3e-5` |
| Budget-sized schedules (Table 3) | `--schedule cosine --min-lr 3e-5 --max-iters 900` to `3600` (2 to 8 epochs), `--eval-interval 150` |
| Corruption-rate sweep (Figure 4) | `--aug-p` from 0.05 to 0.50 |
| Model scale (Table 4) | the 134M architecture above |
| Stronger weight decay (App. C.5) | `--weight-decay 0.5` |

## 4. Loss decomposition on the Markov source (Sec. 5)

Train the data-constrained and the abundant-data runs at a temperature, then:

```bash
python code/decompose.py \
    --constrained runs/mk_t05_none/metrics.jsonl \
    --abundant   runs/mk_t05_abund_none/metrics.jsonl \
    --meta       data_bin_mk_t05/meta.json
```

## 5. Probes on the Markov source (App. C.1)

Training and validation loss on clean windows, the share of 50-token training spans reproduced verbatim (Carlini et al., 2023), the AUC of separating training from validation windows, and the KL divergence of the model's next-token distribution to the true transition and to the marginal:

```bash
python code/probe_memorization.py --ckpt runs/mk_t05_replace/ckpt.pt --data-dir data_bin_mk_t05 --temperature 0.5
```

## 6. Corpus statistics (Sec. 7, App. B.2, B.3, C.3)

```bash
python code/redundancy_stats.py     --bins data_bin_census data_bin_loghub data_bin_roadsafety --sample-tokens 2000000
python code/measure_cond_entropy.py --bins data_bin_census data_bin_loghub --sample-tokens 2000000
python code/redundancy_survey.py    --tokenizer tok_kej_64k.model --manifest survey_manifest.json --out survey.jsonl --raw-bytes 3000000
python code/shrinkage_weight.py     --bin data_bin_koweb/train.bin --vocab 64000 --noise-vocab 64000
python code/shrinkage_weight.py     --bin data_bin_mk_t05/train.bin --vocab 1024 --noise-vocab 1025
```

`survey_manifest.json` lists the 67 surveyed corpora (64 that are not program-generated and 3 program-generated). The raw-text gzip of a trained corpus uses the same script on a manifest entry for its dataset, with `"json_dump": true` for the two JSON corpora.

## 7. Early stopping (App. C.4)

```bash
python code/early_stopping.py --patience 1 3 5 --runs runs/*/metrics.jsonl
```

## Repository layout

```
├── README.md
├── LICENSE               # Apache License 2.0
├── NOTICE
├── requirements.txt
├── survey_manifest.json  # the 67 corpora of the redundancy survey
└── code/                 # flat layout; scripts import model.py and markov_source.py directly
    ├── model.py  train.py  markov_source.py  train_tokenizer.py
    ├── build_hf_corpus.py  build_yodas_transcript.py  build_code_bin.py  build_sql_bin.py  build_loghub_bin.py
    ├── decompose.py  probe_memorization.py  shrinkage_weight.py  early_stopping.py
    └── redundancy_stats.py  redundancy_survey.py  measure_cond_entropy.py
```

## License

Copyright 2026 Flitto Inc. Released under the Apache License 2.0 (see `LICENSE` and `NOTICE`). All corpora are used by reference to their original public sources under their respective licenses.
