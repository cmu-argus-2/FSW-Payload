"""
This is the program that will be running in my computer
when I run the code in rpi it will fisrt check with this server if there are code updates to be done
"""


import socket
import os

HOST = "0.0.0.0"   # Listen on all interfaces
PORT = 5001        # Pick any free port

print(f"Listening on {HOST}:{PORT}")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen()

    while True:
        conn, addr = server.accept()
        with conn:
            print(f"Connected by {addr}")
            data = conn.recv(1024)
            if not data:
                continue
            
            message = data.decode("utf-8")
            print("Received message:", message)

            if message == "sync_now":
                print("  Running remote upload")
                # run remote_upload.sh
                os.system("bash scripts/remote_upload.sh")
            # Optional acknowledgment
            conn.sendall(b"OK")