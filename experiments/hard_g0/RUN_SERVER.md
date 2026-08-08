# HARD-G0 server runbook (physical GPU 0 only)

This executes the preregistered Garden/seed-0 comparison sequentially on one
GPU. It refuses to overwrite any training, evaluation, log, or decision
artifact. A retry therefore requires a fresh `HARD_G0_RUN_SUFFIX`.

## 1. Update and verify the checkout

```bash
cd /home/smbu/dy/mesh-splatting-texel
git pull --ff-only git@github.com:ChizkiyahuOhayon/mesh-splatting-texel.git main
git rev-parse HEAD
git status --short
```

The tracked checkout must be clean. Untracked server-side dataset directories
are allowed.

## 2. Preflight the fixed inputs

```bash
export NAS_ROOT=/home/smbu/dy/nas/meshsplatting_smbu
export GARDEN_DATA=/home/smbu/dy/mesh-splatting/data/mipnerf360/garden
export HARD_G0_RUN_SUFFIX=01

test -d "$GARDEN_DATA/images"
test -f "$GARDEN_DATA/sparse/0/cameras.bin"
test -f "$GARDEN_DATA/sparse/0/images.bin"
test -f "$GARDEN_DATA/sparse/0/points3D.bin"
nvidia-smi -i 0
```

## 3. Run both arms, both evaluations, and the locked decision

```bash
bash experiments/hard_g0/run_gpu0.sh
```

The process is intentionally foregrounded so a failed command is visible and
the shell reports a non-zero status. In another terminal, progress can be
monitored with:

```bash
tail -f "$NAS_ROOT/logs/hard_g0_garden_stock_01.log"
```

After the stock arm completes, switch the filename to
`hard_g0_garden_early25000_01.log`.

## 4. Return the immutable evidence

```bash
cat "$NAS_ROOT/experiments/hard_g0_garden_decision_01.json"
cat "$NAS_ROOT/experiments/hard_g0_garden_stock_01/sac_manifest.json"
cat "$NAS_ROOT/experiments/hard_g0_garden_early25000_01/sac_manifest.json"
```

HARD-G0 passes only if both saved endpoints are numerically `1e-4`, the early
arm gains at least `0.10 dB` PSNR at scaling 4, and its LPIPS at scaling 4 is
non-worse. The script records SSIM but does not use it to change this decision.

If an infrastructure failure leaves partial artifacts, do not delete or reuse
them. Set a fresh suffix (for example `02`) and rerun the same script.
