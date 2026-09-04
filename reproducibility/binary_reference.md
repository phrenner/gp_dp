<!--
┌────────────────────────────────────────────────────────────────────────┐
│  (☁) NUVOLOS (nuvolos.com)                                             │
│  ═════[ DATA ]═════[ ANALYSIS ]═════[ FINDING ]═════[ 10Y+ RE-RUN ]    │
└────────────────────────────────────────────────────────────────────────┘

"Science that can't be re-run is testimony, not evidence."
-->

# Binary Image Reference & Manifest
**Canonical Locator**: `nuvolos/public@sha256:087f6ad182fc08ca19ee6030e659021ab2ee3e8117e0dd64f351267848aec220`  
**Immutable Manifest Digest**: `sha256:087f6ad182fc08ca19ee6030e659021ab2ee3e8117e0dd64f351267848aec220`  
**Image ID**: `sha256:087f6ad182fc08ca19ee6030e659021ab2ee3e8117e0dd64f351267848aec220`  
**Registry**: `docker.io`  
**Architecture / OS**: `linux/amd64`  
**Image Size**: `5.95 GB` (6,393,183,778 bytes)  
**Layer Count**: `17` layers  
**Created**: `2026-08-28T17:38:10.457675544Z`  

## 1. Registry Pull & Verification Command
```bash
docker pull nuvolos/public@sha256:087f6ad182fc08ca19ee6030e659021ab2ee3e8117e0dd64f351267848aec220
```

## 2. Layer Topology (DiffIDs)
| Layer # | DiffID (sha256) |
| :--- | :--- |
| 1 | `sha256:4b7c01ed0534d4f9be9cf97d068da1598c6c20b26cb6134fad066defdb6d541d` |
| 2 | `sha256:83612e3c50f526fc5b5a6a834aa90e5c5154dfb3d9bb4a9e1ce347add70622b3` |
| 3 | `sha256:3dd94d6d02370c6f6efacb719d7aea641026b2694730a833a3f7bc8d540affbd` |
| 4 | `sha256:b3e38b75245668b9f151966cdafd474c9ae5e876666193c1c103a55745ca2960` |
| 5 | `sha256:6b9d2a7e5d8f7124474485c335c84bf1dccb957600062c2be365565866e1e114` |
| 6 | `sha256:aaaa0410a162b6962f5310d365d9f701d70d7d4de109b8ee10efcc9c9421fe35` |
| 7 | `sha256:d68930a4d47465e65da1a17b60ff3d8c210bbf7e84bccff5afb092a5c195db01` |
| 8 | `sha256:fa1d11d9bf7215ea2fadd520b98f1b4c854ed5aafe4a8ebdc2e86bc50c3b2b5e` |
| 9 | `sha256:4159b072156bbad4ffbb5adae24823188880b79811cd0b1726171c7272fe75e0` |
| 10 | `sha256:3d8339fab74c90925feef8868652bdc50b5132a0a1802bc7d5ef6828d1f9ebaa` |
| 11 | `sha256:93a82ff44981f62a2cfe99ea8d54f2e1b3796ecfe7b752f215e965e37daea9ca` |
| 12 | `sha256:2d36b806410087f1ee9ac7f378915f9090d8eee1c980b95408b6f50e8de79719` |
| 13 | `sha256:244464fbe65ccebc9d5ce9a09ca8175f19b1704e048f9b5fd75ebed832096465` |
| 14 | `sha256:92100228d836eddb65bf11ff5448dfb59a2e3346a6b4876ffb64ce80f6c512e5` |
| 15 | `sha256:209a0796b154294158c60ea88233e6cfc2044de3c3e04de49e8cb95503f42afd` |
| 16 | `sha256:c1f3836525ef98e30b03e2c49cce56efc518bf59a41a73648e2e776c2a8de24f` |
| 17 | `sha256:d65605fceba1cb5c45e6c0703c9e04c398a30e2694decedcbf5f2c7e541a0ab1` |

## 3. Runtime Container Configuration
- **User**: `269403658`
- **Working Directory**: `/`
- **Entrypoint**: `["/tini", "-g", "--", "/startup.sh"]`
- **Cmd**: `null`
- **Exposed Ports**: `8888/tcp`
- **Volumes**: `None`
- **Stop Signal**: `SIGTERM`

### Environment Variables
```text
PATH=/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
DEBIAN_FRONTEND=noninteractive
NV_UID=269403658
NV_GID=269400513
DISPLAY=:1
HOME=<REDACTED>
CONDA_DIR=/opt/conda
SHELL=/bin/bash
LC_ALL=en_US.UTF-8
LANG=en_US.UTF-8
LANGUAGE=en_US.UTF-8
TINI_VERSION=v0.19.0
JUPYTER_DATA_DIR=/opt/conda/share/jupyter
JUPYTER_CONFIG_DIR=/opt/conda/etc/jupyter
HOST=<REDACTED>
HPCUSER=<REDACTED>
NV_IS_SHARED=False
NV_IS_STUDENT=False
NVIDIA_DRIVER_CAPABILITIES=compute,utility
NVIDIA_VISIBLE_DEVICES=all
PYTHONPYCACHEPREFIX=/tmp
XDG_CACHE_HOME=/tmp
```

### Image Labels
| Label Key | Value |
| :--- | :--- |
| `NV_APP_VERSION` | `4.3.1` |
| `org.opencontainers.image.ref.name` | `<REDACTED>` |
| `org.opencontainers.image.version` | `<REDACTED>` |
