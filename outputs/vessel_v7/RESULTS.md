# v7.0.0 results

Only the provided 20 labeled training images. No external data or external pretrained weights. All prior versions preserved.

Stage1: scale the saved v6 parameter update relative to its corresponding v4 fold: W(beta)=W_v4+beta*(W_v6-W_v4). Beta grid0/0.25/0.5/0.75/1; this is parameter scaling, not probability blending. Endpoints reuse verified existing held-out probabilities.

Stage2: each rate1e-5 and3e-5 runs all5 folds from the matching v4 checkpoint with newly initialized rank4/alpha4 decoder adapters; 54,720 trainable parameters, 4,416,673 frozen. Train300 batches, batch8, native256 patches, AdamW weight decay1e-4, warmup30, cosine decay, AMP, gradient clip1. Unchanged v4 RGB, sampling, augmentation and loss. Sample seed12026+fold; split seed2026. No extra EMA.

Evaluate D4 at step0 and every25 steps at threshold0.5; include step0 in selection; require improvement exceeding1e-5. Select the best complete OOF candidate at fixed0.5; scores within1e-5 prefer the earlier/simpler candidate. Then select a global threshold from0.40:0.02:0.60.

| Candidate | Fixed0.5 Dice | Precision | Recall | Thin skeleton recall |
|---|---:|---:|---:|---:|
| scale_000 | 0.81370680 | 0.819532 | 0.815043 | 0.718708 |
| scale_025 | 0.81335878 | 0.817043 | 0.817225 | 0.720716 |
| scale_050 | 0.81298361 | 0.814711 | 0.819307 | 0.722675 |
| scale_075 | 0.81232359 | 0.812247 | 0.821059 | 0.724144 |
| scale_100 | 0.81149406 | 0.809730 | 0.822626 | 0.725049 |
| lr_1e-5 | 0.81378070 | 0.819099 | 0.815612 | 0.719359 |
| lr_3e-5 | 0.81391008 | 0.817983 | 0.816996 | 0.720586 |

Selected checkpoint steps:
- lr_1e-5: best steps [225, 0, 0, 200, 0]; 3/5 folds retained step0.
- lr_3e-5: best steps [200, 0, 0, 125, 0]; 3/5 folds retained step0.

Winner: lr_3e-5; threshold 0.5; selected OOF Dice 0.81391008. Final submission_v7.0.0.csv averages the selected family's5 folds with D4 on20 test images.

Selecting checkpoints on each validation fold guarantees a favorable in-sample comparison when step0 is available; it does not establish independent generalization. All scores reuse labels for checkpoint/method/threshold choice, and are not Kaggle scores. Step0-retained folds are not counted as adaptation improvements.

All10 training runs completed. Prior 914 files unchanged. Source hashes, fold isolation, frozen backbones, matching-CUDA merged weights, official conversion and independent20-image RLE decoding verified. White-vessel encoding follows supplied labels and original converter; conflicting prose remains unconfirmed by hidden scoring. No Kaggle upload. All new files on D.
