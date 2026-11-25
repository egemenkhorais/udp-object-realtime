import argparse
import json
from pathlib import Path

import cv2

from src.core.config import load_config
from src.vision.detector import UAVDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Faz 1 — İHA Nesne Tespiti Çalıştırıcı")
    parser.add_argument("--source", required=True, help="Tek görsel yolu veya görsel klasörü")
    parser.add_argument("--weights", default=None, help="Özel .pt ağırlık yolu (opsiyonel)")
    parser.add_argument("--out", default="runs/phase1_demo", help="Çıktı klasörü")
    parser.add_argument("--limit", type=int, default=10, help="Klasör modunda max görsel sayısı")
    parser.add_argument("--dump-json", action="store_true", help="İlk karenin ham JSON çıktısını bas")
    args = parser.parse_args()

    cfg = load_config()
    m_cfg = cfg["model"]
    weights = args.weights or m_cfg["weights"]

    detector = UAVDetector(
        weights=weights,
        device=m_cfg["device"],
        imgsz=m_cfg["imgsz"],
        conf_threshold=m_cfg["conf_threshold"],
        iou_threshold=m_cfg["iou_threshold"],
    )
    print(f"Model yüklendi: {weights} | Cihaz: {detector.device}")
    detector.warmup()

    src_path = Path(args.source)
    image_paths = [src_path] if src_path.is_file() else sorted(src_path.glob("*.jpg"))[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"Görsel bulunamadı: {src_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    latencies: list[float] = []
    for idx, img_path in enumerate(image_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        result = detector.predict(frame, frame_id=idx)
        latencies.append(result.inference_ms)

        annotated = detector.draw(frame, result)
        cv2.imwrite(str(out_dir / img_path.name), annotated)

        print(f"[{idx + 1}/{len(image_paths)}] {img_path.name} -> {len(result.detections)} tespit | {result.inference_ms:.1f} ms")
        if args.dump_json and idx == 0:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    avg_ms = sum(latencies) / len(latencies)
    print(f"\nOrtalama Gecikme: {avg_ms:.2f} ms ({1000.0 / avg_ms:.1f} FPS) | Çıktılar: {out_dir}")


if __name__ == "__main__":
    main()