import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.core.config import resolve_device
from src.core.schemas import Detection, FrameResult


class UAVDetector:
    def __init__(
        self,
        weights: str | Path = "yolo11n.pt",
        device: str = "auto",
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.device = resolve_device(device)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(str(weights))
        self.class_names: dict[int, str] = self.model.names

    def warmup(self) -> None:
        """İlk karedeki model/GPU başlatma gecikmesini ortadan kaldırır."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(
            source=dummy,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

    def predict(self, frame: np.ndarray, frame_id: int = 0) -> FrameResult:
        height, width = frame.shape[:2]
        start = time.perf_counter()

        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        detections: list[Detection] = []

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy_list = boxes.xyxy.cpu().numpy()
            conf_list = boxes.conf.cpu().numpy()
            cls_list = boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy_list, conf_list, cls_list):
                detections.append(
                    Detection(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=self.class_names.get(int(cls_id), str(cls_id)),
                    )
                )

        return FrameResult(
            frame_id=frame_id,
            width=width,
            height=height,
            inference_ms=elapsed_ms,
            detections=detections,
        )

    @staticmethod
    def draw(frame: np.ndarray, result: FrameResult) -> np.ndarray:
        canvas = frame.copy()

        for det in result.detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            color = ((det.class_id * 67) % 255, (det.class_id * 131 + 80) % 255, 220)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.track_id is not None:
                label = f"#{det.track_id} {label}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 2, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )

        hud = f"Frame: {result.frame_id} | Det: {len(result.detections)} | {result.inference_ms:.1f} ms ({result.fps:.1f} FPS)"
        cv2.rectangle(canvas, (8, 8), (470, 36), (20, 20, 20), -1)
        cv2.putText(canvas, hud, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 140), 1, cv2.LINE_AA)
        return canvas