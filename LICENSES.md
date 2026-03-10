# Dependency License Audit

**Project license:** MIT (see `LICENSE`)
**Audit date:** 2026-03-10
**Verdict:** All dependencies are free for commercial use under their stated terms.

---

## Python Packages

| Package | Version pin | SPDX License | Commercial use |
|---|---|---|---|
| fastapi | `>=0.111.0` | MIT | ✅ Free |
| uvicorn | `>=0.29.0` | BSD-3-Clause | ✅ Free |
| pydantic | `>=2.7.0` | MIT | ✅ Free |
| pydantic-settings | `>=2.2.0` | MIT | ✅ Free |
| lxml | `>=5.2.0` | BSD-3-Clause | ✅ Free |
| numpy | `>=1.26.0` | BSD-3-Clause | ✅ Free |
| pandas | `>=2.2.0` | BSD-3-Clause | ✅ Free |
| pyarrow | `>=16.0.0` | Apache-2.0 | ✅ Free |
| scipy | `>=1.13.0` | BSD-3-Clause | ✅ Free |
| scikit-learn | `>=1.4.0` | BSD-3-Clause | ✅ Free |
| torch (PyTorch) | `>=2.3.0` | BSD-3-Clause | ✅ Free |
| mlflow | `>=2.13.0` | Apache-2.0 | ✅ Free |
| celery | `>=5.4.0` | BSD-3-Clause | ✅ Free |
| redis (Python client) | `>=5.0.0` | MIT | ✅ Free |
| loguru | `>=0.7.2` | MIT | ✅ Free |
| pyyaml | `>=6.0.1` | MIT | ✅ Free |
| httpx | `>=0.27.0` | BSD-3-Clause | ✅ Free |
| mcp (Anthropic MCP SDK) | `>=1.0.0` | MIT | ✅ Free |
| anyio | (transitive) | MIT | ✅ Free |
| pytest | `>=7.0` | MIT | ✅ Free (dev only) |
| pytest-asyncio | (dev) | Apache-2.0 | ✅ Free (dev only) |
| pytest-cov | (dev) | MIT | ✅ Free (dev only) |

---

## Docker Images / Runtime Software

| Image / Software | Version | License | Commercial use | Notes |
|---|---|---|---|---|
| `python:3.12-slim` | 3.12 | PSF-2.0 | ✅ Free | Official Python Docker image |
| `redis:7-alpine` | **7.x only** | BSD-3-Clause | ✅ Free | ⚠️ Redis v8+ changed to non-OSI license — stay on v7 |
| `ghcr.io/mlflow/mlflow` | `v2.13.2` | Apache-2.0 | ✅ Free | |
| `mher/flower` | `2.0` | BSD-3-Clause | ✅ Free | |
| `ollama/ollama` | latest | MIT | ✅ Free | |
| `mongo:8.0` | 8.0 | SSPL-1.0 | ✅ Free for internal use | See MongoDB note below |
| `getmeili/meilisearch` | `v1.7.3` | MIT | ✅ Free | Pre-Enterprise split; stay on `<v1.19` for MIT |
| `ghcr.io/danny-avila/librechat` | latest | MIT | ✅ Free | |
| FEBio (`febio4` binary) | 4.x | MIT | ✅ Free | [github.com/febiosoftware/FEBio](https://github.com/febiosoftware/FEBio) |

---

## License Notes

### MongoDB — SSPL-1.0

MongoDB Server Community Edition uses the [Server Side Public License (SSPL) v1](https://www.mongodb.com/legal/licensing/server-side-public-license).

**SSPL does NOT restrict this project's commercial use because:**

- Embedding MongoDB in a proprietary product and selling that product is explicitly allowed.
- Running MongoDB internally (as we do — it is LibreChat's internal datastore) requires no source disclosure.
- The only SSPL restriction applies when you **offer MongoDB itself as a hosted service to external users** (i.e., you are a cloud database provider selling "MongoDB SaaS"). That is not this project.

Reference: [MongoDB SSPL FAQ](https://www.mongodb.com/legal/licensing/server-side-public-license/faq)

---

### Redis — version lock to v7

Redis v7.x is BSD-3-Clause (fully open source).
Redis v8.0+ (released 2024) moved to the [Redis Source Available License (RSAL) + Server Side Public License](https://redis.io/legal/licenses/), which **is not OSI-approved** and restricts cloud-service use.

**Mitigation already in place:** `docker-compose.yml` pins `redis:7-alpine`.
Do **not** upgrade to `redis:8` without legal review.

---

### Meilisearch — version lock to < v1.19

Meilisearch v1.7.3 (used here) is MIT.
Starting from v1.19.0, Meilisearch introduced a dual-license (Community vs Enterprise) and the server binary moved to the [Meilisearch License](https://github.com/meilisearch/meilisearch/blob/main/LICENSE), which restricts managed-service offerings.

**Mitigation already in place:** `docker-compose.librechat.yml` pins `getmeili/meilisearch:v1.7.3`.
Do **not** upgrade past v1.18.x without legal review.

---

### FEBio binary

FEBio is licensed under the MIT License:
[github.com/febiosoftware/FEBio/blob/develop/LICENSE](https://github.com/febiosoftware/FEBio/blob/develop/LICENSE)

FEBio is not bundled in the Docker image (the binary is proprietary to the user's installation). The simulation worker expects it to be bind-mounted or installed in a derived image.

---

## Version Pins That Protect Commercial Use

The following version constraints are **deliberately restrictive** to stay on commercially free builds:

```
# docker-compose.yml
redis:7-alpine          # NOT redis:8 (RSAL license)

# docker-compose.librechat.yml
getmeili/meilisearch:v1.7.3   # NOT v1.19+ (Meilisearch License)
```

If you update either of these, re-run this license audit.
