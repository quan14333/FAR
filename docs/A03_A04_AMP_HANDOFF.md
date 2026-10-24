# Module 1 - A03 AMP / A04 Inference Handoff

## Run identity
- Protocol: inner_clean_v2_v1, Fold 0
- Model: PraNet, channel=32
- Pinned upstream: ad3bf1c51529bda6de50badedba124e1e15a653c
- Checkpoint SHA256: `5268013f6fd90d7c47412875b60b19ef00c37d9de367dd3524cda9368a0c1a61`
- Input: 256 x 256
- AMP: enabled in the new A03 training run
- Best epoch: 25
- Best validation Dice: 0.7348208116683207

## A04 inference
- Status: PASS, 2717/2717 images
- A04 run ID: `a04_fold0_full_5268013f6fd9_20261024T150553407162Z`
- SEG_TRAIN: 1318
- SEG_VAL: 356
- META_FIT: 610
- CALIBRATION: 433
- Probability: float32, sigmoid, native H x W
- Binary threshold: 0.5
- Full independent file audit: PASS (user-observed)
- Kaggle version: interactive-unversioned

## Prediction provenance
- `predictions.csv` is a mapping for external output files.
- All prediction paths are relative to the A04 run output root.
- Actual NPY/PNG files are NOT included in GitHub.
- Filenames use SHA256(image_id), not literal image_id.
- SEG_TRAIN is conservatively labeled `in_sample`;
  it is NOT an OOF prediction set.
- SEG_VAL is validation used during model selection,
  not independent final test.
- No OUTER_TEST or NEVER_POLYP_TEST was evaluated.

## A03 artifact limitation
- `best_model.pth` exists and strict-load passed.
- `last_checkpoint.pth` was NOT found.
- Optimizer/scheduler resume state is not verified.
- The missing resume checkpoint remains an acceptance gap.

## External artifact access
- A03 weights: provide shared artifact location.
- A04 full predictions: provide shared artifact location.
- Verify both links are accessible to Module 2/3 owners.
- Do not mix this run with historical Fold 0 checkpoint.

## Acceptance handoff
1. Download the exact A03 checkpoint and verify SHA256.
2. Load with PraNet(channel=32), strict=True.
3. Run inference on at least 3 images in a fresh session.
4. Check native H x W, probability [0,1], threshold 0.5.
5. Load sample predictions via `predictions.csv`.
6. Confirm all required outputs are accessible.

## Scope
- A03/A04 interim handoff for Module 2/3.
- A05 metrics and A06 visualization are not complete.
- This is not final Module 1 acceptance.
