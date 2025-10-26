import serial 
import time

ser =  serial.Serial('/dev/ttyTHS1', 115200) 
while True: 
    ser.write('H'.encode('utf-8'))
    response = ser.read()
    print("Response:", response.decode('utf-8').strip())
    time.sleep(1)
ser.close()
