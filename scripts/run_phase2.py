import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
import yaml

from src.tracking.tracker import UAVTracker
from src.vision.detector import UAVDetector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_frames(source: Path) -> Tuple[Iterator[Tuple[int, np.ndarray]], float, Tuple[int, int]]:
    """Video dosyası veya sıralı görüntü klasöründen (frame_id, frame) üretir."""
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise FileNotFoundError(f"Klasörde görsel bulunamadı: {source}")
        first = cv2.imread(str(files[0]))
        h, w = first.shape[:2]

        def _gen() -> Iterator[Tuple[int, np.ndarray]]:
            for idx, path in enumerate(files):
                img = cv2.imread(str(path))
                if img is not None:
                    yield idx, img

        return _gen(), 30.0, (w, h)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _gen() -> Iterator[Tuple[int, np.ndarray]]:
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, frame
                idx += 1
        finally:
            cap.release()

    return _gen(), fps, (w, h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: YOLO11n + ByteTrack runner")
    parser.add_argument("--source", type=Path, required=True, help="Video yolu veya sıralı kare klasörü")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--weights", type=str, default=None, help="Config'deki ağırlığı ezmek için")
    parser.add_argument("--device", type=str, default=None, help="mps / cuda / cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/phase2_demo"))
    parser.add_argument("--max-frames", type=int, default=None, help="Hızlı test için kare limiti")
    parser.add_argument("--no-video", action="store_true", help="Video kaydetmeden sadece metrik ölç")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Dedektör kurulumu (mevcut UAVDetector.__init__ imzanla eşleştir)
    det_cfg = cfg.get("detector", cfg.get("model", {}))
    detector = UAVDetector(
        weights=args.weights or det_cfg["weights"],
        device=args.device or det_cfg.get("device", "auto"),
        imgsz=det_cfg.get("imgsz", 960),
        conf_threshold=det_cfg.get("conf_threshold", 0.25),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
    )
    detector.warmup()

    # 2. Tracker kurulumu
    trk_cfg = cfg.get("tracking", {})
    frames_iter, src_fps, (width, height) = iter_frames(args.source)
    tracker = UAVTracker(
        track_activation_threshold=trk_cfg.get("track_activation_threshold", 0.25),
        lost_track_buffer=trk_cfg.get("lost_track_buffer", 30),
        minimum_matching_threshold=trk_cfg.get("minimum_matching_threshold", 0.8),
        frame_rate=int(trk_cfg.get("frame_rate", src_fps)),
        trail_length=trk_cfg.get("trail_length", 20),
    )

    # 3. Video yazıcı (isteğe bağlı)
    writer = None
    out_video_path = args.output / f"{args.source.stem}_tracked.mp4"
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, src_fps, (width, height))

    # 4. Ana akış ve ID kararlılık metrikleri
    track_lifespans = defaultdict(int)  # track_id -> kaç karede yaşadı
    total_frames = 0
    total_detections = 0
    t_pipeline_start = time.perf_counter()

    try:
        for frame_id, frame in frames_iter:
            if args.max_frames is not None and total_frames >= args.max_frames:
                break

            result = detector.predict(frame, frame_id=frame_id)
            result = tracker.update(result)

            total_frames += 1
            total_detections += len(result.detections)
            for det in result.detections:
                if det.track_id is not None:
                    track_lifespans[det.track_id] += 1

            if writer is not None:
                vis = detector.draw(frame, result, trails=tracker.trails)
                writer.write(vis)
    finally:
        if writer is not None:
            writer.release()

    elapsed = max(time.perf_counter() - t_pipeline_start, 1e-6)
    lifespans = list(track_lifespans.values())

    report = {
        "source": str(args.source),
        "frames_processed": total_frames,
        "elapsed_sec": round(elapsed, 3),
        "avg_fps": round(total_frames / elapsed, 2),
        "avg_detections_per_frame": round(total_detections / max(total_frames, 1), 2),
        "unique_track_ids": len(lifespans),
        "avg_track_lifespan_frames": round(sum(lifespans) / max(len(lifespans), 1), 2),
        "max_track_lifespan_frames": max(lifespans, default=0),
    }

    report_path = args.output / "phase2_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    if writer is not None:
        print(f"Çıktı video: {out_video_path}")


if __name__ == "__main__":
    main()