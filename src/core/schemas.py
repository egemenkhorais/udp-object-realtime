from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None  # Faz 2 (ByteTrack) için hazır alan

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class FrameResult:
    frame_id: int
    width: int
    height: int
    inference_ms: float
    detections: list[Detection] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return 1000.0 / self.inference_ms if self.inference_ms > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "width": self.width,
            "height": self.height,
            "inference_ms": round(self.inference_ms, 2),
            "fps": round(self.fps, 1),
            "count": len(self.detections),
            "detections": [asdict(d) for d in self.detections],
        }