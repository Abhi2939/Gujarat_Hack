# Project NETRA — Implementation Status and Handover

## 1. Purpose

NETRA is a proof-of-concept AI perception pipeline for disconnected CCTV
cameras. It converts raw frames into searchable vehicle observations, plate
reads, vehicle attributes, Re-ID embeddings, and watchlist alerts.

The intended end-to-end path is:

```text
Video / RTSP frame
  -> YOLO vehicle detection
  -> ByteTrack within-camera tracking
  -> plate detection + OCR
  -> colour + body-type fingerprint
  -> OSNet vehicle Re-ID embedding
  -> PostgreSQL/pgvector + Redis Streams
  -> correlation and watchlist alerting
```

## 2. Implemented components

| Area | Location | Status |
|---|---|---|
| Detection | `Detection_Tracking/detector.py` | YOLO vehicle/person detection implemented. |
| Tracking | `Detection_Tracking/tracker.py` | ByteTrack IDs with reconnect and scene-cut resets. |
| Camera/video frame bridge | `worker.py` | Reads a local video, webcam, or RTSP source and passes frames to L2. |
| L2 orchestrator | `Detection_Tracking/end_to_end.py` | Connects tracking, plate OCR, attributes, Re-ID, storage, and alerting. |
| Plate detection | `Plate_OCR/plate_detector.py` | Configurable YOLO plate-model adapter. |
| Plate OCR | `Plate_OCR/plate_ocr.py` | PaddleOCR adapter, multiline handling, and conditional deblur retry. |
| Plate association | `Plate_OCR/plate_pipeline.py` | Maps plate reads back to the tracked vehicle and global frame coordinates. |
| Vehicle fingerprint | `Vehicle_Fingerprint/` | HSV/K-Means colour and rule-based body type. |
| Re-ID | `Re_ID/` | OSNet embedding pipeline, cosine matcher, and pgvector query path. |
| Storage | `Storage_Events/` | PostgreSQL tables, pgvector, optional TimescaleDB, Redis Streams publishers. |
| Correlation | `correlation_alerting/` | Exact/fuzzy plate, Re-ID, and attributes candidate matching. |
| Watchlist alerts | `correlation_alerting/` | Stores and publishes exact/fuzzy plate-watchlist alerts. |
| Movement analytics | `Movement_Analytics/` | Plate movement history and Markov next-camera prediction. |
| Optional night vision | `Night Vision/` | Brightness gate and enhancement modules; not yet wired into the worker. |

## 3. Repairs completed during this implementation pass

- Added the missing plate detector and full plate-to-track pipeline.
- Repaired missing OCR protocols/helpers/imports and the blur-module name mismatch.
- Added `PaddleOCRBackend` behind a lazy import so code can be tested without loading OCR models.
- Added `EndToEndPerceptionPipeline` to connect formerly independent L2 modules.
- Added `worker.py` to decode a video/RTSP stream instead of discarding frames.
- Added Redis publishing for `watchlist.alert` events.
- Fixed the fuzzy plate SQL query to return `track_session_id` and `track_id`, as required by correlation.
- Added a pgvector cosine-search index to the database schema.
- Added unit tests for plate association, orchestration, alert publishing, and fuzzy-query fields.
- Updated `ML_FLOW.md` to distinguish implemented code from external deployment prerequisites.

## 4. Run the local video pipeline

Activate the virtual environment and run the supplied test video:

```powershell
python worker.py --source .\14739260_1080_1920_60fps.mp4 --camera-id test-video-01 --max-frames 120 --every-n 3
```

Expected log shape:

```text
frame=3 tracks=<count> plates=0 attributes=<count> embeddings=0 alerts=0
```

Without optional weights, this runs detection, tracking, and vehicle
fingerprinting. `plates=0` and `embeddings=0` are expected in this mode.

### Enable plate detection/OCR

```powershell
python worker.py --source .\14739260_1080_1920_60fps.mp4 --camera-id test-video-01 --plate-weights .\models\indian_plate.pt
```

The supplied `indian_plate.pt` must be a trained plate-detector checkpoint.

### Enable vehicle Re-ID

```powershell
python worker.py --source .\14739260_1080_1920_60fps.mp4 --camera-id test-video-01 --reid-weights .\models\osnet_vehicle.pth
```

Only use a trained vehicle Re-ID checkpoint. An OSNet model with no vehicle
weights does not produce reliable cross-camera matches.

## 5. Tests

Run:

```powershell
python -m pytest -q
```

The test suite currently contains four unit tests. The first observed run had
three passes and one failure caused by a fake database cursor requiring an
unnecessary `params` argument. That test stub has been fixed; rerun the command
above in the active virtual environment to record the final result.

## 6. Database and event services

The storage layer expects:

- PostgreSQL with the `pgvector` extension.
- Redis 7+ for event streams.
- TimescaleDB is optional; the schema falls back to ordinary PostgreSQL tables.

Initialize the schema after the services are available:

```powershell
python -c "from Storage_Events.db import connect, init_schema; c=connect('postgresql://postgres:netra@localhost:5432/netra'); init_schema(c); print('schema ready')"
```

Pass a `StoragePipeline` and optionally an `AlertingPipeline` to
`EndToEndPerceptionPipeline` from the deployment bootstrap to persist events
and publish Redis messages.

## 7. Event types

| Event | Meaning |
|---|---|
| `vehicle.tracked` | One detector/tracker observation. |
| `plate.read` | One OCR read associated with a tracked vehicle. |
| `vehicle.attributes` | Colour and body-type estimate. |
| `vehicle.reid_embedding` | L2-normalized vehicle appearance vector. |
| `watchlist.alert` | Exact or fuzzy plate watchlist match. |

## 8. What remains

### Required for a demo with real cameras

1. Re-run and pass the test suite after the test-stub repair.
2. Supply a trained Indian plate detector and Indian HSRP-aware PaddleOCR model.
3. Supply a trained vehicle OSNet/Re-ID checkpoint.
4. Start and configure PostgreSQL/pgvector plus Redis.
5. Connect the actual L1 camera ingestion/reconnect service to `worker.py` or
   directly to `EndToEndPerceptionPipeline`.
6. Test against representative daytime, night, moving, occluded, and two-wheeler footage.
7. Tune confidence thresholds and frame sampling rate for the available CPU/GPU.

### Optional enhancement work

- Wire the existing night-vision modules into the worker behind a brightness threshold.
- Replace the heuristic body type with a trained lightweight classifier.
- Add super-resolution/deblur model backends after validating their latency benefit.
- Add Docker Compose/deployment configuration for PostgreSQL, Redis, and the worker.
- Add integration tests using real Redis/PostgreSQL containers and sample footage.

## 9. Important operating constraints

- YOLO/Ultralytics is AGPL-3.0; assess licensing before commercial deployment.
- Re-ID matches are review candidates only and must not independently raise automatic alerts.
- OCR accuracy figures depend on the supplied model and footage; accuracy has not yet been benchmarked on Gujarat camera feeds.
- Face recognition and thermal/IR processing are not enabled in this worker.

## 10. Repository state

`ML_FLOW.md` is the ML design document. This file is the implementation and
handover snapshot. Model checkpoints, camera credentials, database credentials,
and benchmark datasets are intentionally not stored in the repository.
