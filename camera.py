import argparse
import time
from typing import Optional, Tuple

import cv2
from ultralytics import YOLO

# Helper module that handles MQTT connection and publishing.
from relay_mqtt import MqttRelayPublisher, add_mqtt_arguments, env_bool


# In COCO-trained YOLO models, class ID 0 is "person".
PERSON_CLASS_ID = 0


def parse_camera_source(value: str):
    """
    Converts the camera argument into the correct OpenCV source type.

    If the user passes "0", "1", etc., it becomes an integer webcam index.
    If the user passes a URL like rtsp://..., it stays a string.
    """
    if value.isdigit():
        return int(value)

    return value


def build_parser() -> argparse.ArgumentParser:
    """
    Builds command-line arguments for the YOLO camera program.
    """
    parser = argparse.ArgumentParser(
        description="Run YOLO person detection and publish relay commands over MQTT."
    )

    # YOLO model file. yolo11n.pt is the small/nano model.
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path.")

    # Camera source. Usually 0 for default webcam.
    parser.add_argument(
        "--camera",
        default="0",
        help="OpenCV camera source. Use 0 for the default webcam or an RTSP/HTTP stream URL.",
    )

    # Minimum confidence required before a person detection counts.
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Minimum YOLO confidence for person detection.",
    )

    # Number of missed frames before changing human_present to False.
    # This helps prevent flickering if YOLO misses a person for one frame.
    parser.add_argument(
        "--missing-frames",
        type=int,
        default=10,
        help="Number of consecutive missed frames before publishing relay off.",
    )

    # Sends a heartbeat even when the state does not change.
    # This lets the Pi know the camera system is still alive.
    parser.add_argument(
        "--publish-interval",
        type=float,
        default=1.0,
        help="Heartbeat publish interval in seconds while the state is unchanged.",
    )

    # Disable OpenCV preview window for headless use.
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=not env_bool("DISPLAY_VIDEO", True),
        help="Disable the OpenCV preview window.",
    )

    # Dry run prints MQTT payloads instead of sending them.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print MQTT payloads instead of sending them.",
    )

    # Adds MQTT options like broker, topic, token, username, password, TLS, etc.
    add_mqtt_arguments(parser)

    return parser


def detect_people(
    frame,
    model: YOLO,
    confidence_threshold: float,
    draw: bool,
) -> Tuple[bool, Optional[float], int]:
    """
    Runs YOLO on one frame and checks for people.

    Returns:
    - person_detected: True if at least one person is found
    - best_confidence: highest confidence score among detected people
    - detected_people: total number of people detected
    """
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

            # Ignore anything that is not a person.
            if class_id != PERSON_CLASS_ID or confidence < confidence_threshold:
                continue

            person_detected = True
            detected_people += 1

            # Track the highest confidence score.
            best_confidence = confidence if best_confidence is None else max(best_confidence, confidence)

            # Draw bounding boxes only if the preview window is enabled.
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
    """
    Draws the current human detection and MQTT connection state on the preview window.
    """
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
    """
    Main YOLO camera loop.

    This script:
    1. Opens the camera.
    2. Runs YOLO detection.
    3. Decides whether a human is present.
    4. Publishes the human_present state to MQTT.
    """
    args = build_parser().parse_args()

    # Display window is enabled unless --no-display is used.
    display = not args.no_display

    # Create MQTT publisher using arguments from relay_mqtt.py.
    publisher = MqttRelayPublisher.from_args(args)
    publisher.connect()

    # Load YOLO model.
    model = YOLO(args.model)

    # Open camera or stream source.
    cap = cv2.VideoCapture(parse_camera_source(args.camera))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {args.camera}")

    # Current stable human state.
    human_present = False

    # Last state successfully published to MQTT.
    last_published_state: Optional[bool] = None

    # Time of last successful publish.
    last_publish_at = 0.0

    # Start missing frame counter at the threshold.
    # This means human_present will start as False until a person is detected.
    missing_frames = args.missing_frames

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Camera frame read failed; stopping.")
                break

            # Run YOLO and detect people in the current frame.
            raw_detected, best_confidence, detected_people = detect_people(
                frame,
                model,
                args.confidence,
                draw=display,
            )

            # Stabilize detection:
            # If a person is detected, immediately set human_present True.
            # If not detected, require multiple missed frames before setting False.
            if raw_detected:
                human_present = True
                missing_frames = 0
            else:
                missing_frames += 1

                if missing_frames >= args.missing_frames:
                    human_present = False

            now = time.monotonic()

            # Publish immediately if the state changed.
            state_changed = human_present != last_published_state

            # Also publish periodically as a heartbeat.
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

            # Show preview window if enabled.
            if display:
                draw_status(frame, human_present, publisher.status)
                cv2.imshow("YOLO Person Detection", frame)

                # Press q to quit.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Stopping after keyboard interrupt.")

    finally:
        # On shutdown, publish human_present=False for safety.
        publisher.publish_relay_state(
            human_present=False,
            confidence=None,
            detected_people=0,
            reason="shutdown",
            wait=True,
        )

        # Release camera and close window.
        cap.release()

        if display:
            cv2.destroyAllWindows()

        # Disconnect MQTT.
        publisher.close()


if __name__ == "__main__":
    main()
