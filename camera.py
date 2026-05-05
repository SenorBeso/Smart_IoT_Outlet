import argparse
import time
from typing import Optional, Tuple

import cv2
from ultralytics import YOLO

from relay_mqtt import MqttRelayPublisher, add_mqtt_arguments, env_bool


PERSON_CLASS_ID = 0


def parse_camera_source(value: str):
    """Allow webcam indexes like 0 and stream URLs like rtsp://..."""
    if value.isdigit():
        return int(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO person detection and publish relay commands over MQTT."
    )
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path.")
    parser.add_argument(
        "--camera",
        default="0",
        help="OpenCV camera source. Use 0 for the default webcam or an RTSP/HTTP stream URL.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Minimum YOLO confidence for person detection.",
    )
    parser.add_argument(
        "--missing-frames",
        type=int,
        default=10,
        help="Number of consecutive missed frames before publishing relay off.",
    )
    parser.add_argument(
        "--publish-interval",
        type=float,
        default=1.0,
        help="Heartbeat publish interval in seconds while the state is unchanged.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=not env_bool("DISPLAY_VIDEO", True),
        help="Disable the OpenCV preview window.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print MQTT payloads instead of sending them.",
    )
    add_mqtt_arguments(parser)
    return parser


def detect_people(
    frame,
    model: YOLO,
    confidence_threshold: float,
    draw: bool,
) -> Tuple[bool, Optional[float], int]:
    results = model(frame, verbose=False)
    person_detected = False
    best_confidence: Optional[float] = None
    detected_people = 0

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            if class_id != PERSON_CLASS_ID or confidence < confidence_threshold:
                continue

            person_detected = True
            detected_people += 1
            best_confidence = confidence if best_confidence is None else max(best_confidence, confidence)

            if draw:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"person {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    return person_detected, best_confidence, detected_people


def draw_status(frame, human_present: bool, mqtt_status: str) -> None:
    status_text = f"human_present = {human_present} | mqtt = {mqtt_status}"
    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if human_present else (0, 0, 255),
        2,
    )


def main() -> None:
    args = build_parser().parse_args()
    display = not args.no_display

    publisher = MqttRelayPublisher.from_args(args)
    publisher.connect()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(parse_camera_source(args.camera))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {args.camera}")

    human_present = False
    last_published_state: Optional[bool] = None
    last_publish_at = 0.0
    missing_frames = args.missing_frames

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame read failed; stopping.")
                break

            raw_detected, best_confidence, detected_people = detect_people(
                frame,
                model,
                args.confidence,
                draw=display,
            )

            if raw_detected:
                human_present = True
                missing_frames = 0
            else:
                missing_frames += 1
                if missing_frames >= args.missing_frames:
                    human_present = False

            now = time.monotonic()
            state_changed = human_present != last_published_state
            heartbeat_due = now - last_publish_at >= args.publish_interval

            if state_changed or heartbeat_due:
                published = publisher.publish_relay_state(
                    human_present=human_present,
                    confidence=best_confidence,
                    detected_people=detected_people,
                    reason="yolo",
                )
                if published:
                    last_published_state = human_present
                    last_publish_at = now
                    confidence_text = "none" if best_confidence is None else f"{best_confidence:.2f}"
                    print(
                        f"human_present={human_present} "
                        f"confidence={confidence_text} "
                        f"people={detected_people} "
                        f"mqtt={publisher.status}"
                    )

            if display:
                draw_status(frame, human_present, publisher.status)
                cv2.imshow("YOLO Person Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Stopping after keyboard interrupt.")
    finally:
        publisher.publish_relay_state(
            human_present=False,
            confidence=None,
            detected_people=0,
            reason="shutdown",
            wait=True,
        )
        cap.release()
        if display:
            cv2.destroyAllWindows()
        publisher.close()


if __name__ == "__main__":
    main()
