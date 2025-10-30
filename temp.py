import serial 
import time

ser =  serial.Serial('/dev/ttyTHS1', 115200, timeout=1) 

time.sleep(2)

ser.write(b"Hello world\m")
print("Sent message to device")

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode("utf-8", errors="ignore")
        print(f"Received: {line}")
        ser.write(b"Hi")
