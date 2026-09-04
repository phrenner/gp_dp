<!--
┌────────────────────────────────────────────────────────────────────────┐
│  (☁) NUVOLOS (nuvolos.com)                                             │
│  ═════[ DATA ]═════[ ANALYSIS ]═════[ FINDING ]═════[ 10Y+ RE-RUN ]    │
└────────────────────────────────────────────────────────────────────────┘

"Science that can't be re-run is testimony, not evidence."
-->

# Reproducibility & Build Guide

## Source Artifact Details
- **Image Target**: `nuvolos/public@sha256:087f6ad182fc08ca19ee6030e659021ab2ee3e8117e0dd64f351267848aec220`
- **Synthesized Base**: `ubuntu:24.04`
- **Total Inferred Instructions**: 41

## Reconstruction & Build Context Recommendations
1. **External Context Artifacts**: The original build referenced external files copied via `COPY` or `ADD`:
   - `/start_app.sh`
   - `/usr/local/etc/odbc.ini`
   - `/`
   - `/bashrc_template`
   - `/etc/odbc.ini`
   - `/etc/odbcinst.ini`
   - `/usr/local/etc/odbcinst.ini`

   *Ensure appropriate placeholder or recovered configuration files exist in your local build context before executing `docker build`.*

## Point-in-Time Reproduction Limitations
- **Package Index Volatility**: Upstream package repositories (APT, Conda, PyPI, NPM) evolve over time. Unless strict lockfiles or pinned snapshot archives are used, minor sub-dependencies may resolve to newer patch versions.
- **Build Non-Determinism**: Timestamps, file inode ordering, and environment variables prevent bit-for-bit identical layer sha256 checksums.
- **Authoritative Binary**: For production execution requiring exact bit-level consistency, always use the canonical image binary digest reference provided in `binary_reference.md`.

## Build Invocation
```bash
docker build -t nuvolos/public:reproduced -f Dockerfile.reconstructed .
```
