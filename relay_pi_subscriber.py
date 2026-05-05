import argparse
import csv
import json
import os
import signal
import sqlite3
import statistics
import time
from datetime import datetime

import paho.mqtt.client as mqtt
from gpiozero import OutputDevice

# Adafruit/CircuitPython libraries for Raspberry Pi I2C and ADS1115 ADC.
# board and busio come from adafruit-blinka.
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


# Default MQTT topic.
# The Windows YOLO computer publishes messages to this topic.
# The Raspberry Pi subscribes to this topic.
DEFAULT_TOPIC = "smartlab/relay/command"


def env_bool(name: str, default: bool) -> bool:
    """
    Read a boolean value from an environment variable.

    This allows settings such as:
        RELAY_ACTIVE_LOW=true

    Accepted true values:
        1, true, yes, on
    """
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    """
    Build all command-line options for the subscriber.

    Most values can be set either by command-line arguments or environment
    variables. This makes testing and deployment easier.
    """
    parser = argparse.ArgumentParser(
        description="Relay Pi subscriber with YOLO MQTT, SCT power sensing, CSV logging, SQLite logging, and latched shutdown."
    )

    # MQTT broker settings.
    # Since Mosquitto usually runs on the Pi itself, localhost is the default.
    parser.add_argument("--mqtt-broker", default=os.getenv("MQTT_BROKER_HOST", "localhost"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_BROKER_PORT", "1883")))
    parser.add_argument("--mqtt-topic", default=os.getenv("MQTT_RELAY_TOPIC", DEFAULT_TOPIC))
    parser.add_argument("--mqtt-client-id", default=os.getenv("MQTT_CLIENT_ID", "relay-pi-node"))

    # Optional MQTT username/password.
    # These are not required if the broker allows anonymous local connections.
    parser.add_argument("--mqtt-username", default=os.getenv("MQTT_USERNAME"))
    parser.add_argument("--mqtt-password", default=os.getenv("MQTT_PASSWORD"))

    # Shared token used to reject unauthorized MQTT messages.
    parser.add_argument("--mqtt-token", default=os.getenv("MQTT_SHARED_TOKEN"))

    # GPIO relay settings.
    # This script uses BCM numbering.
    # Physical pin 12 on the Raspberry Pi is BCM GPIO18.
    parser.add_argument("--relay-pin", type=int, default=int(os.getenv("RELAY_PIN", "18")))

    # Some relay modules are active-low.
    # If the relay acts backwards, run with --active-low.
    parser.add_argument("--active-low", action="store_true", default=env_bool("RELAY_ACTIVE_LOW", False))

    # If no valid YOLO update is received within this many seconds,
    # the Pi assumes no human is present.
    parser.add_argument("--human-timeout", type=float, default=float(os.getenv("HUMAN_TIMEOUT_SECONDS", "5")))

    # RMS voltage threshold for deciding whether power is being drawn.
    parser.add_argument("--power-threshold", type=float, default=float(os.getenv("POWER_THRESHOLD", "0.0002")))

    # Sampling settings for the SCT current sensor.
    parser.add_argument("--sample-count", type=int, default=int(os.getenv("SCT_SAMPLE_COUNT", "100")))
    parser.add_argument("--sample-delay", type=float, default=float(os.getenv("SCT_SAMPLE_DELAY", "0.002")))

    # Delay between each control loop iteration.
    parser.add_argument("--loop-delay", type=float, default=float(os.getenv("CONTROL_LOOP_DELAY", "0.25")))

    # Local CSV and SQLite logging files.
    parser.add_argument("--log-file", default=os.getenv("POWER_LOG_FILE", "power_log.csv"))
    parser.add_argument("--database-file", default=os.getenv("DATABASE_FILE", "iot_history.db"))

    return parser


class PowerSensor:
    """
    Reads the SCT current transformer through an ADS1115 ADC.

    The SCT signal is AC, but the ADC cannot read negative voltage.
    The circuit biases the signal around a midpoint, usually around 1.65V.
    This class subtracts the midpoint and calculates RMS to estimate
    AC signal strength.
    """

    def __init__(self, sample_count: int, sample_delay: float) -> None:
        self.sample_count = sample_count
        self.sample_delay = sample_delay

        # Open I2C bus using the Pi's SCL and SDA pins.
        i2c = busio.I2C(board.SCL, board.SDA)

        # Initialize the ADS1115 ADC.
        self.ads = ADS.ADS1115(i2c)

        # Gain controls the ADC input range.
        self.ads.gain = 1

        # Read from ADS1115 channel A0.
        self.chan = AnalogIn(self.ads, 0)

    def read_rms(self):
        """
        Read multiple voltage samples and calculate:
            midpoint: the DC bias voltage
            rms: the AC RMS voltage after subtracting the midpoint
        """
        samples = []

        # Collect voltage samples from the ADC.
        for _ in range(self.sample_count):
            samples.append(self.chan.voltage)
            time.sleep(self.sample_delay)

        # The midpoint is the average of the biased waveform.
        midpoint = statistics.mean(samples)

        # Remove the midpoint so the remaining signal is AC movement.
        centered = [v - midpoint for v in samples]

        # Calculate RMS of the centered AC signal.
        rms = (sum(v * v for v in centered) / len(centered)) ** 0.5

        return midpoint, rms


class RelayController:
    """
    Controls the relay and stores the most recent human detection state.

    The final safety decision is made in the main loop.
    """

    def __init__(self, relay_pin: int, active_low: bool) -> None:
        # Initialize relay output pin.
        # active_high is flipped when using an active-low relay board.
        self.relay = OutputDevice(
            relay_pin,
            active_high=not active_low,
            initial_value=True,
        )

        # Track current relay state so changes can be printed clearly.
        self.current_state = True

        # Most recent valid YOLO human state.
        self.human_present = False

        # Time of the last valid YOLO update.
        self.last_human_update_at = 0.0

        # Start the relay on when the program starts.
        self.set_relay(True, "startup")

    def update_human_present(self, value: bool) -> None:
        """
        Store the latest human detection state received through MQTT.
        """
        self.human_present = value
        self.last_human_update_at = time.monotonic()

    def get_effective_human(self, timeout: float) -> bool:
        """
        Return the current human state if updates are recent.

        If the YOLO workstation stops sending updates, assume no human
        is present. This is a safety fallback.
        """
        if time.monotonic() - self.last_human_update_at > timeout:
            return False

        return self.human_present

    def set_relay(self, state: bool, reason: str) -> None:
        """
        Set relay state.

        state=True  means relay ON.
        state=False means relay OFF.
        """
        if state:
            self.relay.on()
        else:
            self.relay.off()

        # Only print when the relay state actually changes.
        if state != self.current_state:
            print(f"Relay {'ON' if state else 'OFF'} ({reason})")

        self.current_state = state

    def close(self) -> None:
        """
        Turn off relay and release GPIO resources during shutdown.
        """
        self.set_relay(False, "shutdown")
        self.relay.close()


class CsvLogger:
    """
    Logs each reading to a CSV file.

    This is useful for quick viewing in Excel or including sample data
    in the final project report.
    """

    def __init__(self, filename: str) -> None:
        file_exists = os.path.exists(filename)

        # Append mode preserves old readings.
        self.file = open(filename, "a", newline="")
        self.writer = csv.writer(self.file)

        # If the file is new, write the header.
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
        """
        Write one row to the CSV file and flush immediately.
        """
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        """
        Close the CSV file.
        """
        self.file.close()


class DatabaseLogger:
    """
    Logs each reading to a local SQLite database.

    This gives the project a real local backend for historical analysis.
    The database file defaults to:
        iot_history.db
    """

    def __init__(self, database_file: str) -> None:
        self.connection = sqlite3.connect(database_file)
        self.cursor = self.connection.cursor()

        # Create table if it does not already exist.
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                midpoint_v REAL NOT NULL,
                rms_v REAL NOT NULL,
                power_drawn INTEGER NOT NULL,
                human_present INTEGER NOT NULL,
                relay_on INTEGER NOT NULL,
                safety_latched INTEGER NOT NULL
            )
        """)

        self.connection.commit()

    def write(
        self,
        timestamp: str,
        midpoint: float,
        rms: float,
        power_drawn: bool,
        human_present: bool,
        relay_on: bool,
        safety_latched: bool,
    ) -> None:
        """
        Insert one sensor/control reading into SQLite.

        Boolean values are stored as 0 or 1.
        """
        self.cursor.execute("""
            INSERT INTO sensor_log (
                timestamp,
                midpoint_v,
                rms_v,
                power_drawn,
                human_present,
                relay_on,
                safety_latched
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            midpoint,
            rms,
            int(power_drawn),
            int(human_present),
            int(relay_on),
            int(safety_latched),
        ))

        self.connection.commit()

    def close(self) -> None:
        """
        Close the SQLite database connection.
        """
        self.connection.close()


def parse_human(payload: dict):
    """
    Extract the human detection state from an MQTT payload.

    Preferred payload format:
        {"human_present": true}

    relay_on and relay are also supported so the manual test script can
    still be used.
    """
    if "human_present" in payload:
        return bool(payload["human_present"])

    if "relay_on" in payload:
        return bool(payload["relay_on"])

    if "relay" in payload:
        return str(payload["relay"]).strip().upper() == "ON"

    return None


def new_mqtt_client(client_id: str):
    """
    Create a Paho MQTT client.

    This supports both newer and older versions of paho-mqtt.
    New versions need CallbackAPIVersion.VERSION2.
    """
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except (AttributeError, TypeError):
        return mqtt.Client(client_id=client_id)


def main() -> None:
    """
    Main Raspberry Pi program.

    The Pi:
    1. Subscribes to YOLO MQTT messages.
    2. Reads the SCT current sensor.
    3. Calculates RMS voltage.
    4. Applies relay safety logic.
    5. Logs to CSV and SQLite.
    """
    args = build_parser().parse_args()

    # Initialize hardware and loggers.
    controller = RelayController(args.relay_pin, args.active_low)
    sensor = PowerSensor(args.sample_count, args.sample_delay)
    csv_logger = CsvLogger(args.log_file)
    db_logger = DatabaseLogger(args.database_file)

    # Once this becomes True, the relay stays off until program restart.
    safety_latched = False

    # Used for clean program shutdown.
    stop_requested = False

    def stop(sig, frame):
        nonlocal stop_requested
        stop_requested = True

    # Support Ctrl+C and system stop signals.
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def on_connect(client, userdata, flags, reason_code, properties=None):
        """
        MQTT callback that runs after connecting to the broker.
        """
        if hasattr(reason_code, "is_failure") and reason_code.is_failure:
            print(f"MQTT connection failed: {reason_code}")
            return

        if reason_code not in (0, "Success") and str(reason_code).lower() not in {"0", "success"}:
            print(f"MQTT connection failed: {reason_code}")
            return

        print(f"Connected to MQTT broker {args.mqtt_broker}:{args.mqtt_port}")

        # Subscribe to YOLO command topic.
        client.subscribe(args.mqtt_topic, qos=1)
        print(f"Subscribed to {args.mqtt_topic}")

    def on_message(client, userdata, msg):
        """
        MQTT callback that runs when a message arrives.

        It validates JSON, checks the shared token, and updates the
        human_present state.
        """
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            print("Rejected invalid JSON command.")
            return

        # Security check: reject messages without the correct token.
        if args.mqtt_token and payload.get("token") != args.mqtt_token:
            print("Rejected command with missing or invalid token.")
            return

        human = parse_human(payload)

        if human is None:
            print("Rejected command without human_present state.")
            return

        # Save the new YOLO state.
        controller.update_human_present(human)
        print(f"YOLO update: human_present={human}")

    # Create MQTT client and attach callbacks.
    client = new_mqtt_client(args.mqtt_client_id)
    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    # Optional username/password if broker authentication is enabled.
    if args.mqtt_username:
        client.username_pw_set(args.mqtt_username, args.mqtt_password)

    print(f"Relay GPIO BCM {args.relay_pin}; active_low={args.active_low}")
    print(f"Power threshold: {args.power_threshold:.4f} V RMS")
    print(f"CSV logging to: {args.log_file}")
    print(f"SQLite logging to: {args.database_file}")

    # Connect to MQTT broker and begin background MQTT loop.
    client.connect(args.mqtt_broker, args.mqtt_port, keepalive=30)
    client.loop_start()

    try:
        while not stop_requested:
            # Read current sensor and calculate RMS.
            midpoint, rms = sensor.read_rms()

            # Convert RMS voltage into a boolean power-drawn state.
            power_drawn = rms > args.power_threshold

            # Get current human state, falling back to False if YOLO timed out.
            human_present = controller.get_effective_human(args.human_timeout)

            # Final relay safety logic.
            if safety_latched:
                # Once latched, stay off until restart.
                controller.set_relay(False, "latched safety shutdown")

            elif power_drawn and not human_present:
                # Unsafe condition:
                # Something is drawing current while no human is detected.
                safety_latched = True
                controller.set_relay(False, "SAFETY LATCH TRIGGERED")

            elif not human_present:
                # Normal no-human condition.
                # Shut off relay but do not latch unless power is drawn.
                controller.set_relay(False, "no human detected")

            else:
                # Human is present and no safety latch has triggered.
                controller.set_relay(True, "human detected")

            # Timestamp for console, CSV, and SQLite.
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Print live state to console for debugging/demo.
            print(
                f"[{timestamp}] "
                f"midpoint={midpoint:.3f}V | "
                f"rms={rms:.4f}V | "
                f"power_drawn={power_drawn} | "
                f"human_present={human_present} | "
                f"relay_on={controller.current_state} | "
                f"safety_latched={safety_latched}"
            )

            # Build CSV row.
            row = [
                timestamp,
                f"{midpoint:.3f}",
                f"{rms:.4f}",
                power_drawn,
                human_present,
                controller.current_state,
                safety_latched,
            ]

            # Log to CSV.
            csv_logger.write(row)

            # Log to SQLite database.
            db_logger.write(
                timestamp,
                midpoint,
                rms,
                power_drawn,
                human_present,
                controller.current_state,
                safety_latched,
            )

            time.sleep(args.loop_delay)

    finally:
        # Cleanup always runs, even after Ctrl+C.
        client.loop_stop()
        client.disconnect()
        csv_logger.close()
        db_logger.close()
        controller.close()


if __name__ == "__main__":
    main()
