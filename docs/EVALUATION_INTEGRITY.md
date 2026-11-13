# Medical-imaging evaluation and inference integrity

The inference script now requires an existing compatible checkpoint, and the evaluator rejects unavailable/incompatible checkpoints, unreadable images and empty evaluation datasets. Previous behavior could report random-weight predictions and could silently substitute zero images.

**These safeguards do not establish medical usefulness.** The evaluator currently uses the same colposcopy input for both encoder branches when histopathology data is unavailable. This is a proxy input arrangement, not a genuine paired-modal evaluation. Do not treat results from that proxy as dual-modal diagnostic performance. The SWede-to-binary label mapping also requires dataset-owner validation and cannot alone establish a cancer diagnosis.

Before reporting results, use patient-level separation, verified ground-truth labels, independent cohorts, calibrated uncertainty, explicit exclusion counts, checkpoint provenance and documented ethics/consent. Never present model confidence as a clinical probability without calibration evidence.
