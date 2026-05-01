# VoiceEmo — Speech-Based Emotion Recognition

NLP703 (Speech Processing) project, MBZUAI · April 2026 · Karim Mahfouz.

A comparative study of self-supervised speech encoders (Wav2Vec2, HuBERT, WavLM) on RAVDESS, with multi-seed verification, layer-wise probing, and zero-shot cross-corpus evaluation on CREMA-D.

## Headline findings

- **Encoder ranking is fragile.** Under three random seeds per encoder and matched hyperparameters, Wav2Vec2 (0.751 ± 0.042), HuBERT (0.729 ± 0.022), and WavLM (0.742 ± 0.015) cluster within seed variance.
- **Mid-layer probing beats fine-tuning.** A frozen linear probe at WavLM layer 5 reaches 0.767 — above the fully fine-tuned model — with ~15,000× fewer trainable parameters.
- **In-domain accuracy overstates real performance.** RAVDESS-trained models lose ~30 absolute accuracy points zero-shot on CREMA-D, with a structural neutral → sad collapse (717 of 1,087 utterances) shared across all three encoders.

## Repository layout

```
src/
├── data_loader.py      RAVDESS dataset, manifest parser, speaker-independent splits
├── model.py            SSL encoder + mean-pool + classifier head
├── train.py            Training loop with seed control
├── evaluate.py         Confusion matrices, per-class metrics, plots
├── predict.py          Single-clip inference
├── mfcc_baseline.py    MFCC + logistic regression baseline
├── layer_probe.py      Frozen layer-wise linear probing
└── cross_corpus.py     Zero-shot RAVDESS → CREMA-D evaluation

requirements.txt        Pinned dependencies (transformers==4.38.2 etc.)
LICENSE                 MIT
```

## Reproducing the results

```bash
pip install -r requirements.txt
# (PyTorch installed separately per pytorch.org)
python -m src.train --model wav2vec2-base --seed 42
python -m src.layer_probe --model wavlm-base
python -m src.cross_corpus --train-corpus ravdess --test-corpus crema-d
```

The full pipeline runs in ~4 hours on a single A100 80 GB. Speaker-independent splits (actors 1–16 train, 17–20 val, 21–24 test) are enforced in `data_loader.py`.

## Citation context

This is a course project; please cite the underlying datasets and models if you use this code:

- Livingstone & Russo (2018), RAVDESS · PLoS ONE
- Cao et al. (2014), CREMA-D · IEEE TAC
- Baevski et al. (2020), wav2vec 2.0 · NeurIPS
- Hsu et al. (2021), HuBERT · TASLP
- Chen et al. (2022), WavLM · JSTSP

## License

MIT (see `LICENSE`).
