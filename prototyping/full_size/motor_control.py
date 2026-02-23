import serial
import time

# CONFIGURATION
# Replace with the port you found in Step 2
SERIAL_PORT = '/dev/ttyACM0'  
# Replace with the baud rate specified in your motor controller manual
BAUD_RATE = 115200 

def init_control():
	try:
		# 1. Open the Serial Connection
		# timeout=1 means "wait 1 second for a message, then give up" so we don't freeze
		ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
		
		# Wait a moment for the connection to settle
		while not ser.is_open:
			time.sleep(2) 
		print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
		command = "S\n" 
		ser.write(command.encode('utf-8'))
		print(f"Sent: {command.strip()}")
		return ser
	except serial.SerialException as e:
		print(f"Error: Could not open serial port {SERIAL_PORT}. Is it plugged in?")
		print(f"Details: {e}")
		return None
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
		return None

def kick(ser, com):
	try:
		# 2. SENDING A MESSAGE
		# Motor controllers often expect a command ending in a newline (\n) or carriage return (\r)
		# The .encode('utf-8') converts your string into bytes
		command = f"{com}\n" 
		ser.write(command.encode('utf-8'))
		print(f"Sent: {command.strip()}")

		# 4. Close the connection when done
		# ser.close()
	except serial.SerialException as e:
		print(f"Error: Could not open serial port {SERIAL_PORT}. Is it plugged in?")
		print(f"Details: {e}")
		return None
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
		return None

def receive(ser):
	while True:
		try:
			if ser.in_waiting > 0:
				# Read the line until a \n is found
				response = ser.readline()
				
				# Decode bytes back into a string and strip whitespace
				decoded_response = response.decode('utf-8').strip()
				print(f"Received: {decoded_response}")
		except serial.SerialException as e:
			print(f"Error: Could not open serial port {SERIAL_PORT}. Is it plugged in?")
			print(f"Details: {e}")
		except Exception as e:
			print(f"An unexpected error occurred: {e}")
