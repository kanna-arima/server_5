import csv
import time
import board
import digitalio
from adafruit_mcp3xxx.mcp3008 import MCP3008
from adafruit_mcp3xxx.analog_in import AnalogIn

# Initialize SPI and MCP3008
spi = board.SPI()
cs = digitalio.DigitalInOut(board.D8)  # Chip Select pin (CE0)
mcp = MCP3008(spi, cs)

# Setup potentiometer channel (Channel 0)
pot_channel = AnalogIn(mcp, 0)

# CSV File Setup
csv_file = '/home/pi4/Desktop/server_5/data/sensor.csv'

# Open the CSV file in append mode to allow real-time updates
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Potentiometer Value", "Voltage"])  # Headers

    print("Realtime potentiometer reading. Press Ctrl+C to stop.\n")
    print(f"{'Time':<20}{'Value':<10}{'Voltage (V)':<10}")
    print("-" * 40)

    try:
        while True:
            # Read potentiometer data
            value = pot_channel.value
            voltage = pot_channel.voltage
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            # Log to CSV
            writer.writerow([timestamp, value, voltage])
            file.flush()  # Ensure data is written to the file immediately

            # Display in real-time
            print(f"{timestamp:<20}{value:<10}{voltage:<10.2f}")

            # Delay between readings
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped by Sensei! Realtime updates halted.")
