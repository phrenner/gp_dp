# Docker image and environment archive

This folder documents the Docker image in which the replication package was developed and validated, and explains how to run the replication inside it. The image is **not required** to replicate the paper; see the [main README](../README.md) for the native installation (Step 1) and the replication workflows (Steps 2-5).

The image is identified by its digest:

```bash
nuvolos/public@sha256:087f6ad182fc08ca19ee6030e659021ab2ee3e8117e0dd64f351267848aec220
```

## 1. Contents of this folder

| File(s) | What it contains |
| ------- | ---------------- |
| `binary_reference.md`, `manifest.json` | Image metadata: digest, size, the 17 layer hashes, and the runtime configuration (user, entrypoint, environment variables). `manifest.json` is the machine-readable version of `binary_reference.md`. |
| `sbom.json`, `dependencies.md`, `dependencies.json` | Software bill of materials: an inventory of all 1,614 packages installed in the image (OS packages, Conda, pip, NPM, Go), with versions and licenses. `sbom.json` is in SPDX 2.3 format; `dependencies.md` is the human-readable table. |
| `Dockerfile.reconstructed`, `Dockerfile.reconstructed.appended`, `delta_reconciliation.json`, `REPRODUCIBILITY.md` | Documentation of how the image was built. `Dockerfile.reconstructed` is the base build recipe inferred from the image layers; `Dockerfile.reconstructed.appended` adds the changes made interactively afterwards (creation of the `rep` conda environment, package installation), which are also listed in `delta_reconciliation.json`. `REPRODUCIBILITY.md` lists the files from the original build that are no longer available. |
| `SUMMARY.md` | One-page overview of the three groups above. |

The reconstructed Dockerfiles are documentation only. Because some files from the original build are unavailable, rebuilding from them produces a similar, but not byte-identical, image. Always use the pinned digest above as the reference environment.

## 2. Running the replication inside the container

Requirements: Docker on a Linux/amd64 host, about 6 GB of disk space for the image, plus the unpacked replication package.

```bash
# 1. Go to the folder containing the unpacked replication package
#    (the folder with README.md, requirements.txt, 1_install.sh, ..., 5_run_all_models.sh)
cd PATH_TO_REPLICATION

# 2. Pull the image by digest
docker pull nuvolos/public@sha256:087f6ad182fc08ca19ee6030e659021ab2ee3e8117e0dd64f351267848aec220

# 3. Start an interactive shell in the container, with the package mounted at /files
docker run --rm -it \
  -v "$(pwd):/files" \
  -w /files \
  --entrypoint /bin/bash \
  nuvolos/public@sha256:087f6ad182fc08ca19ee6030e659021ab2ee3e8117e0dd64f351267848aec220
```

Inside the container, the conda environment `rep` from Step 1 of the main README is already installed, so `1_install.sh` does not need to be run. Activate it and run the replication scripts exactly as described in the main README:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate rep

./2_replicate_original.sh    # Step 2: figures and tables from the deposited checkpoints (~15 minutes)
```

Steps 3-5 (`3_replicate_fp.sh`, `4_validate_BE.sh`, `5_run_all_models.sh`) are run the same way; their runtime and hardware requirements are given in the main README. Output is written to the mounted folder and therefore remains on the host after the container exits.

## 3. Rebuilding the image (optional)

To rebuild an approximation of the image from the reconstructed recipe:

```bash
docker build -t nuvolos/public:reproduced -f Dockerfile.reconstructed .
```

`REPRODUCIBILITY.md` lists the files referenced by the original build that must be supplied (or replaced by placeholders) in the build context. The result will not be byte-identical to the pinned image; use the digest in Section 2 for any check that depends on the exact environment.
