from collections import defaultdict, deque
from typing import Dict, Deque, Tuple
import numpy as np
import supervision as sv

from src.core.schemas import Detection, FrameResult


class UAVTracker:
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        trail_length: int = 20,
    ) -> None:
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        self.trail_length = trail_length
        self.trails: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.trail_length)
        )

    def update(self, frame_result: FrameResult) -> FrameResult:
        if not frame_result.detections:
            # Boş karede de tracker'ı güncelle ki kayıp sayaçları ilerlesin
            self.tracker.update_with_detections(sv.Detections.empty())
            return frame_result

        xyxy = np.array(
            [[d.x1, d.y1, d.x2, d.y2] for d in frame_result.detections],
            dtype=np.float32,
        )
        confidence = np.array(
            [d.confidence for d in frame_result.detections],
            dtype=np.float32,
        )
        class_id = np.array(
            [d.class_id for d in frame_result.detections],
            dtype=np.int32,
        )

        # Sınıf adlarını map'te tut (ByteTrack sonrası eşlemek için)
        class_names = {d.class_id: d.class_name for d in frame_result.detections}

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )
        tracked = self.tracker.update_with_detections(sv_dets)

        tracked_detections = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i].tolist()
            cid = int(tracked.class_id[i])
            conf = float(tracked.confidence[i])
            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None

            if tid is not None:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                self.trails[tid].append((cx, cy))

            tracked_detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    class_id=cid,
                    class_name=class_names.get(cid, str(cid)),
                    track_id=tid,
                )
            )

        frame_result.detections = tracked_detections
        return frame_result

    def reset(self) -> None:
        self.tracker.reset()
        self.trails.clear()