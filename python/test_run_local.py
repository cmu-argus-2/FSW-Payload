import time 
from controller import PayloadController
from ipc_comms import PayloadIPC


if __name__ == '__main__':

    ipc = PayloadIPC()
    controller = PayloadController()
    controller.initialize(ipc)

    # make sure the connection is established
    resp = controller.communication_interface.is_connected()

    if resp:
        print("[INFO] Connection established.")
    else:
        print("[ERROR] Connection failed.")
        exit(1)

    NUM_PINGS = 3
    counter = 0

    for i in range(1, NUM_PINGS+1):
        print("[INFO] Sending ping...")

        resp = controller.ping()
        time.sleep(0.1)

        if resp:
            counter += 1
            print("[INFO] Ping succeeded.")
        else:
            print("[ERROR] Ping failed.")

        time.sleep(0.5)
        print(f"[INFO] {counter}/{i} ping(s) succeeded.")

    print("[INFO] Done.")
    controller.deinitialize()
    exit(0)