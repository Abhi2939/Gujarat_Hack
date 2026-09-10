# ML_FLOW.md — Project NETRA: AI Perception Pipeline (Phase 3 / L2)

> Consolidated from project docs (`AIModelStack.md`, `ML-CV-Stack-Recommendations.md`) and working discussion. This is the ML/CV side of NETRA — turning raw camera frames into structured, searchable vehicle facts.

---

## 1. Where this fits in the bigger picture

NETRA connects Gujarat's ~80,000 disconnected CCTV cameras (26 departments, incompatible systems) into one platform that can track a vehicle across camera/department boundaries and alert an officer in real time. The hackathon PoC scope: ~50 sandbox cameras (10 confirmed live), mock watchlist.

Pipeline: `L0 Registry → L1 Ingestion → L2 AI Perception (this doc) → L3 Event Bus → L4 Correlation/Alerting → L5 Data Layer → L6 Command Centre → L7 External Integrations`

Status: the L2 software pipeline is implemented as composable stages. Production-quality
ANPR and Re-ID still require approved, trained model weights plus reachable Postgres and
Redis services; those are deployment inputs, not code that can be inferred from this repo.

---

## 2. The corrected ML pipeline (step by step)

```
Frame → [1] Detect (YOLOv8) → [2] Track within-camera (ByteTrack)
      → [3] Plate detect + OCR (YOLOv8 plate head → PaddleOCR)
      → [4] Vehicle fingerprint (colour + body-type)
      → [5] Re-ID embedding (OSNet) → cosine similarity across cameras
      → [6] Store (TimescaleDB + pgvector) → publish events (Redis Streams)
```

### Step 1 — Object Detection (YOLOv8)
Draws a box around every vehicle/person in each frame. Runs on **every frame, every camera** — must be fast. Use `yolov8n` (CPU/PoC) or `yolov8s` (GPU) — not the largest variant.
- COCO classes usable directly: `car`, `motorcycle`, `bus`, `truck`, `person`
- Auto-rickshaws/3-wheelers likely misclassify as car/motorcycle → fine-tune on an Indian traffic dataset if this proves insufficient

### Step 2 — Tracking (ByteTrack, via `supervision`)
Gives each detected vehicle a **stable ID within one continuous camera feed** — so "this box in frame 1" and "this box in frame 2" are recognized as the same physical vehicle as it moves.
- **Scope: single camera only.** The ID does *not* carry over when the vehicle leaves the frame and appears on a different camera — that's a new, unrelated track.
- Resets on reconnect *or* in-stream scene-cut (ADR-007) so looping/cut footage can't fake a continuous trajectory.

### Step 3 — Plate Detection + OCR
On the tracked vehicle's crop: a second YOLOv8 model (fine-tuned) finds the plate region → crop → PaddleOCR (PP-OCRv5, Awiros Indian fine-tune) reads the text.
- Pretrained OCR without Indian fine-tuning: ~58% accuracy. Fine-tuned: ~98%.
- Must handle multi-line HSRP plates (common on 2-wheelers) — either a line-split-aware detector or a single/two-line classifier before OCR.
- This produces plate **text**, not a "tracked number" — it's a per-sighting read, not a persistent ID.

### Step 4 — Vehicle Fingerprint (colour + body-type)
Cheap backup descriptor computed alongside plate reading — used when the plate can't be read (very common: angle, blur, occlusion, dirty plate).
- Colour: OpenCV HSV histogram + K-Means (not a deep model — overkill for 8 colour classes)
- Body-type: rule-based on bounding-box aspect ratio, or a lightweight MobileNet classifier

### Step 5 — Re-ID (OSNet) — cross-camera matching
**This is the actual gap** current design (plate match + attribute fallback) doesn't cover: recognizing "same car" across *different* cameras when the plate was never read at either one.
- OSNet turns each vehicle crop into a fixed-length embedding vector (e.g. 512-dim) — a learned "appearance fingerprint," robust to viewpoint/angle changes
- Trained via triplet loss: same-vehicle embeddings pulled close together, different-vehicle embeddings pushed apart — it never learns "this is a Swift," just "produce a discriminative fingerprint"
- "Omni-scale": parallel conv streams at different receptive-field sizes + a learned gate that weighs which scale matters per image (a scratch/sticker vs. overall silhouette)
- Lightweight (depthwise-separable convs) — built for exactly this kind of real-time, many-camera deployment
- Matching: cosine similarity between two embeddings; use pretrained VeRi-776 weights via `torchreid`, consider fine-tuning later on real sandbox footage
- **Confidence caveat**: less reliable than plate match — treat as "possible match for review," not an automatic alert trigger, at least for the demo

### Step 6 — Store + Publish
- Plate text, colour, body-type → TimescaleDB (`plate_reads`, `vehicle_tracks`)
- Re-ID embedding → **pgvector** (Postgres extension, `vector(512)` column — reuses existing DB, no new infra)
- Structured events (`plate.read`, `vehicle.attributes`) → Redis Streams (PoC stand-in for Kafka) for Phase 4 (Correlation/Alerting) to consume

---

## 3. Matching tiers this feeds into (Phase 4 / L4)

```
1. Exact plate match        → sub-100ms, highest confidence
2. Fuzzy plate match        → Levenshtein ≤2, absorbs OCR error
3. Re-ID embedding match     → NEW — handles unreadable plates (this doc's Step 5)
4. Colour + body-type only  → lowest confidence, analyst review
```

---

## 4. Full ML capability roster (from AIModelStack.md)

| # | Capability | Model / Approach | Tier | Notes |
|---|---|---|---|---|
| 1 | ANPR | YOLO plate head → PaddleOCR/Parseq (Indian HSRP fine-tune) | MVP | cheap, small crop |
| 2 | Object detection | YOLOv8 s/m (Indian traffic classes) | MVP | runs every frame, INT8-quantize |
| 3 | Face detection/recognition | SCRFD/RetinaFace-mobile + ArcFace | Bonus/gated | RBAC + approval only, never general scan |
| 4 | Night vision | Training augmentation; optional Zero-DCE++ | Enhancement | conditional, brightness-triggered |
| 5 | Heat/UV/IR | `sensor_type` routing hook only | Hardware-gated | no real thermal cameras confirmed |
| 6 | Blur handling | Blur-augmented training; NAFNet-small retry | MVP(aug)/Enhancement | retry only on low OCR confidence |
| 7 | Pixel reconstruction (SR) | Lightweight SR (Real-ESRGAN distilled/FSRCNN) | Enhancement | plate-crop only, conditional |
| 8 | Route prediction | Markov transition matrix | MVP | statistical, no GPU needed |
| 9 | Multi-camera ID (Re-ID) | OSNet + pgvector/FAISS | MVP-adjacent (new scope) | **this doc's focus** |
| 10 | Movement history | SQL query | Deterministic | no model |
| 11 | Searchable events | Postgres + `pg_trgm` | Deterministic | no model |

**Scaling principle**: always-on (detection/tracking/ANPR) vs. conditionally-triggered (deblur, SR, night-enhance, face — cheap gate first) vs. query-time (route prediction, search — no inference cost). This is what makes it viable from 50 → 80,000 cameras on an unresolved GPU budget.

---

## 5. Recommended open-source stack

| Component | Library | License | Install |
|---|---|---|---|
| Detection | ultralytics (YOLOv8/11/26) | AGPL-3.0 | `pip install ultralytics` |
| Tracking | supervision (ByteTrack) | MIT | `pip install supervision` |
| Plate detection | YOLOv8 fine-tuned (Indian plates) | MIT | load `.pt` weights |
| Plate OCR | PaddleOCR PP-OCRv5 (Awiros fine-tune) | Apache 2.0 | `pip install paddleocr` |
| Colour | OpenCV HSV + K-Means | — | `opencv-python`, `scikit-learn` |
| Re-ID | torchreid (OSNet) | — | pretrained VeRi-776 checkpoint |
| Vector store | pgvector | — | Postgres extension |
| Event bus | Redis Streams (PoC) / Kafka (prod) | — | already in stack |

**Avoid**: YOLO-NAS (less maintained), DeepSORT (slower than ByteTrack), Tesseract (fails on Indian plates), pretrained PaddleOCR without fine-tuning (58% vs 98% accuracy), `pip install yolov8` (wrong package — use `ultralytics`).

---

## 6. Build order (implementation status)

- [x] L0 Registry, L1 Ingestion — done, verified against sandbox
- [x] **Step 1-2 — Detection + Tracking**, including scene-cut/reconnect resets
- [x] Step 3 — plate-detection/OCR pipeline and multiline/deblur retry wiring
- [x] Step 4 — vehicle fingerprint (colour + rule-based body type)
- [x] Step 5 — OSNet embedding and cosine/pgvector matching wiring
- [x] Step 6 — Postgres/pgvector persistence, Redis Streams publishing, and an end-to-end orchestrator
- [x] Phase 4 — exact/fuzzy plate matching, Re-ID/attribute candidates, and watchlist alert wiring

### Deployment prerequisites (not code-complete claims)

- Supply the Indian plate-detector weights and the approved PaddleOCR fine-tune to `YOLOPlateDetector` and `PaddleOCRBackend`.
- Supply a VeRi-776/vehicle-trained OSNet checkpoint to `OSNetBackend(weights_path=...)`; embeddings produced without one must not be matched.
- Initialise Postgres with pgvector (and optionally TimescaleDB) and Redis, then provide their connections to `StoragePipeline` / `AlertingPipeline`.
- Feed decoded camera frames into `EndToEndPerceptionPipeline`; it composes tracking, OCR, attributes, Re-ID, storage, and alerting in that order.

### Custom dev effort estimates
| Component | Effort |
|---|---|
| Frame bridge (ffmpeg → numpy) — worker.py currently discards frames | 1-2 days |
| Indian plate fine-tuning (~500-1000 annotated images) | 2-3 days |
| `plate.read` / `vehicle.attributes` event schemas | 0.5-1 day |
| Redis Streams wiring for L3 | 1 day |
| Watchlist matching (Phase 4) | 3-5 days |

---

## 7. Open risks to keep in mind

- No GPU on hackathon dev machine → use `yolov8n`, process every Nth frame, reduce resolution
- Re-ID accuracy is viewpoint-sensitive — front vs. rear shots of the same car are genuinely hard to match; frame it as "assists when plates fail," not guaranteed cross-camera tracking
- AGPL-3.0 on ultralytics — fine for hackathon, needs Enterprise License if this goes commercial
- Camera ingestion must deliver decoded frames to `EndToEndPerceptionPipeline`; the source/worker integration is deployment-specific and is not present in this repository.
