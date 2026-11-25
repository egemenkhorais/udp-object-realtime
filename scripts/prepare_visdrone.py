import shutil
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from src.core.config import load_config

VALID_VISDRONE_CATEGORIES = set(range(1, 11))  # 1..10 -> YOLO 0..9


def convert_split(raw_split_dir: Path, out_images_dir: Path, out_labels_dir: Path) -> dict[str, int]:
    src_images = raw_split_dir / "images"
    src_annots = raw_split_dir / "annotations"

    if not src_images.exists() or not src_annots.exists():
        raise FileNotFoundError(f"Eksik klasör yapısı: {raw_split_dir} (images/ ve annotations/ olmalı)")

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "valid_boxes": 0, "filtered_boxes": 0}
    image_paths = sorted(src_images.glob("*.jpg"))

    for img_path in tqdm(image_paths, desc=f"Dönüştürülüyor: {raw_split_dir.name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        annot_path = src_annots / f"{img_path.stem}.txt"
        yolo_lines: list[str] = []

        if annot_path.exists():
            with annot_path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    parts = raw_line.strip().rstrip(",").split(",")
                    if len(parts) < 8:
                        continue

                    x, y, w, h = map(float, parts[:4])
                    score = int(parts[4])
                    category = int(parts[5])

                    # Ignored regions (score=0 veya category=0) ve 'others' (11) atılır
                    if score == 0 or category not in VALID_VISDRONE_CATEGORIES:
                        stats["filtered_boxes"] += 1
                        continue

                    # Görüntü dışına taşan kutuları sınırlara kırp
                    x1 = max(0.0, min(x, float(img_w)))
                    y1 = max(0.0, min(y, float(img_h)))
                    x2 = max(0.0, min(x + w, float(img_w)))
                    y2 = max(0.0, min(y + h, float(img_h)))

                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 1.0 or bh <= 1.0:
                        stats["filtered_boxes"] += 1
                        continue

                    xc = (x1 + bw / 2.0) / img_w
                    yc = (y1 + bh / 2.0) / img_h
                    nw = bw / img_w
                    nh = bh / img_h
                    cls_id = category - 1

                    yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                    stats["valid_boxes"] += 1

        # Görüntüyü kopyala (diskten tasarruf etmek istersen shutil.copy2 yerine symlink kullanabilirsin)
        dst_img = out_images_dir / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        dst_label = out_labels_dir / f"{img_path.stem}.txt"
        dst_label.write_text("\n".join(yolo_lines), encoding="utf-8")
        stats["images"] += 1

    return stats


def main() -> None:
    cfg = load_config()
    raw_root = Path(cfg["dataset"]["raw_root"])
    processed_root = Path(cfg["dataset"]["processed_root"]).resolve()

    summary: dict[str, dict[str, int]] = {}
    for split_name, folder_name in cfg["dataset"]["splits"].items():
        raw_split_dir = raw_root / folder_name
        if not raw_split_dir.exists():
            print(f"[ATLANDI] Klasör bulunamadı: {raw_split_dir}")
            continue

        out_img = processed_root / "images" / split_name
        out_lbl = processed_root / "labels" / split_name
        summary[split_name] = convert_split(raw_split_dir, out_img, out_lbl)

    dataset_yaml = {
        "path": str(processed_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": cfg["dataset"]["classes"],
    }

    yaml_out = Path(cfg["dataset"]["yaml_path"])
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    with yaml_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)

    print("\n--- Dönüşüm Raporu ---")
    for split, st in summary.items():
        print(f"{split:>5} | Görsel: {st['images']:>5} | Geçerli Kutu: {st['valid_boxes']:>7} | Elenen: {st['filtered_boxes']:>6}")
    print(f"YOLO YAML oluşturuldu: {yaml_out}")


if __name__ == "__main__":
    main()