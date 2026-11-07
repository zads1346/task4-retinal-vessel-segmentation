# v5.0.0 results

Selected experiment: A. All new files on D; all prior versions preserved.

A screening Dice: 0.823262; B: 0.822158.

Full OOF candidate Dice at 0.5: 0.816022. v4: 0.813707.

Final candidate/v4 weights: 1/0; threshold 0.5; tuned OOF Dice 0.816022.

submission_v5.0.0.csv is the OOF-selected primary file. submission_equal_blend/submission.csv is the 50:50 alternative at the primary threshold, not separately optimized.

These are selection-biased validation scores, not Kaggle scores. Thin skeleton is a digital-label proxy. E/IterNet informs attention to continuity only; no iterative architecture is claimed or added to A/B.

Independent audit passed 84837 checks. No upload performed.

## Requested v4 integration

Use submission_v5.0.0_ensemble.csv for the requested genuine blend: A 75% + v4 25%, threshold 0.5. OOF Dice 0.81602112. Unconstrained selection chose A alone by a negligible fixed-threshold margin and is retained in submission_v5.0.0.csv. Both outputs passed official conversion and independent pixelwise RLE decoding.
