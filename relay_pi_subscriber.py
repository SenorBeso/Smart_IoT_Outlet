import argparse
import csv
import json
import os
import signal
import statistics
import time
from datetime import datetime

import paho.mqtt.client as mqtt
from gpiozero import OutputDevice

import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


DEFAULT_TOPIC = "smartlab/relay/command"


def env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relay Pi subscriber with YOLO MQTT, SCT power sensing, and latched shutdown."
    )

    parser.add_argument("--mqtt-broker", default=os.getenv("MQTT_BROKER_HOST", "localhost"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_BROKER_PORT", "1883")))
    parser.add_argument("--mqtt-topic", default=os.getenv("MQTT_RELAY_TOPIC", DEFAULT_TOPIC))
    parser.add_argument("--mqtt-client-id", default=os.getenv("MQTT_CLIENT_ID", "relay-pi-node"))
    parser.add_argument("--mqtt-username", default=os.getenv("MQTT_USERNAME"))
    parser.add_argument("--mqtt-password", default=os.getenv("MQTT_PASSWORD"))
    parser.add_argument("--mqtt-token", default=os.getenv("MQTT_SHARED_TOKEN"))

    parser.add_argument("--relay-pin", type=int, default=int(os.getenv("RELAY_PIN", "18")))
    parser.add_argument("--active-low", action="store_true", default=env_bool("RELAY_ACTIVE_LOW", False))

    parser.add_argument("--human-timeout", type=float, default=float(os.getenv("HUMAN_TIMEOUT_SECONDS", "5")))
    parser.add_argument("--power-threshold", type=float, default=float(os.getenv("POWER_THRESHOLD", "0.030")))

    parser.add_argument("--sample-count", type=int, default=int(os.getenv("SCT_SAMPLE_COUNT", "100")))
    parser.add_argument("--sample-delay", type=float, default=float(os.getenv("SCT_SAMPLE_DELAY", "0.002")))
    parser.add_argument("--loop-delay", type=float, default=float(os.getenv("CONTROL_LOOP_DELAY", "0.25")))

    parser.add_argument("--log-file", default=os.getenv("POWER_LOG_FILE", "power_log.csv"))

    return parser


class PowerSensor:
    def __init__(self, sample_count: int, sample_delay: float) -> None:
        self.sample_count = sample_count
        self.sample_delay = sample_delay

        i2c = busio.I2C(board.SCL, board.SDA)
        self.ads = ADS.ADS1115(i2c)
        self.ads.gain = 1
        self.chan = AnalogIn(self.ads, 0)

    def read_rms(self):
        samples = []

        for _ in range(self.sample_count):
            samples.append(self.chan.voltage)
            time.sleep(self.sample_delay)

        midpoint = statistics.mean(samples)
        centered = [v - midpoint for v in samples]
        rms = (sum(v * v for v in centered) / len(centered)) ** 0.5

        return midpoint, rms


class RelayController:
    def __init__(self, relay_pin: int, active_low: bool) -> None:
        self.relay = OutputDevice(
            relay_pin,
            active_high=not active_low,
            initial_value=True,
        )

        self.current_state = True
        self.human_present = False
        self.last_human_update_at = 0.0

        self.set_relay(True, "startup")

    def update_human_present(self, value: bool) -> None:
        self.human_present = value
        self.last_human_update_at = time.monotonic()

    def get_effective_human(self, timeout: float) -> bool:
        if time.monotonic() - self.last_human_update_at > timeout:
            return False
        return self.human_present

    def set_relay(self, state: bool, reason: str) -> None:
        if state:
            self.relay.on()
        else:
            self.relay.off()

        if state != self.current_state:
            print(f"Relay {'ON' if state else 'OFF'} ({reason})")

        self.current_state = state

    def close(self) -> None:
        self.set_relay(False, "shutdown")
        self.relay.close()


class Logger:
    def __init__(self, filename: str) -> None:
        file_exists = os.path.exists(filename)
        self.file = open(filename, "a", newline="")
        self.writer = csv.writer(self.file)

        if not file_exists:
            self.writer.writerow([
                "timestamp",
                "midpoint_v",
                "rms_v",
                "power_drawn",
                "human_present",
                "relay_on",
                "safety_latched",
            ])
            self.file.flush()

    def write(self, row) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def parse_human(payload: dict):
    if "human_present" in payload:
        return bool(payload["human_present"])

    if "relay_on" in payload:
        return bool(payload["relay_on"])

    if "relay" in payload:
        return str(payload["relay"]).strip().upper() == "ON"

    return None


def new_mqtt_client(client_id: str):
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except (AttributeError, TypeError):
        return mqtt.Client(client_id=client_id)


def main() -> None:
    args = build_parser().parse_args()

    controller = RelayController(args.relay_pin, args.active_low)
    sensor = PowerSensor(args.sample_count, args.sample_delay)
    logger = Logger(args.log_file)

    safety_latched = False
    stop_requested = False

    def stop(sig, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if hasattr(reason_code, "is_failure") and reason_code.is_failure:
            print(f"MQTT connection failed: {reason_code}")
            return

        if reason_code not in (0, "Success") and str(reason_code).lower() not in {"0", "success"}:
            print(f"MQTT connection failed: {reason_code}")
            return

        print(f"Connected to MQTT broker {args.mqtt_broker}:{args.mqtt_port}")
        client.subscribe(args.mqtt_topic, qos=1)
        print(f"Subscribed to {args.mqtt_topic}")

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            print("Rejected invalid JSON command.")
            return

        if args.mqtt_token and payload.get("token") != args.mqtt_token:
            print("Rejected command with missing or invalid token.")
            return

        human = parse_human(payload)

        if human is None:
            print("Rejected command without human_present state.")
            return

        controller.update_human_present(human)
        print(f"YOLO update: human_present={human}")

    client = new_mqtt_client(args.mqtt_client_id)
    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    if args.mqtt_username:
        client.username_pw_set(args.mqtt_username, args.mqtt_password)

    print(f"Relay GPIO BCM {args.relay_pin}; active_low={args.active_low}")
    print(f"Power threshold: {args.power_threshold:.4f} V RMS")
    print(f"Logging to: {args.log_file}")

    client.connect(args.mqtt_broker, args.mqtt_port, keepalive=30)
    client.loop_start()

    try:
        while not stop_requested:
            midpoint, rms = sensor.read_rms()
            power_drawn = rms > args.power_threshold
            human_present = controller.get_effective_human(args.human_timeout)

            if safety_latched:
                controller.set_relay(False, "latched safety shutdown")

            elif power_drawn and not human_present:
                safety_latched = True
                controller.set_relay(False, "SAFETY LATCH TRIGGERED")

            elif not human_present:
                controller.set_relay(False, "no human detected")

            else:
                controller.set_relay(True, "human detected")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(
                f"[{timestamp}] "
                f"midpoint={midpoint:.3f}V | "
                f"rms={rms:.4f}V | "
                f"power_drawn={power_drawn} | "
                f"human_present={human_present} | "
                f"relay_on={controller.current_state} | "
                f"safety_latched={safety_latched}"
            )

            logger.write([
                timestamp,
                f"{midpoint:.3f}",
                f"{rms:.4f}",
                power_drawn,
                human_present,
                controller.current_state,
                safety_latched,
            ])

            time.sleep(args.loop_delay)

    finally:
        client.loop_stop()
        client.disconnect()
        logger.close()
        controller.close()


if __name__ == "__main__":
    main()
