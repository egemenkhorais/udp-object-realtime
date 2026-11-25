from ultralytics import YOLO

from src.core.config import load_config, resolve_device


def main() -> None:
    cfg = load_config()
    m_cfg = cfg["model"]
    t_cfg = cfg["train"]

    device = resolve_device(m_cfg["device"])
    print(f"Eğitim başlıyor | Base: {m_cfg['weights']} | Cihaz: {device}")

    model = YOLO(m_cfg["weights"])
    model.train(
        data=cfg["dataset"]["yaml_path"],
        epochs=t_cfg["epochs"],
        imgsz=m_cfg["imgsz"],
        batch=t_cfg["batch_size"],
        workers=t_cfg["workers"],
        device=device,
        project=t_cfg["project_dir"],
        name=t_cfg["run_name"],
        seed=cfg["project"]["seed"],
        exist_ok=True,
    )


if __name__ == "__main__":
    main()