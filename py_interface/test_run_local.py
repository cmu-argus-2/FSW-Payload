import time 
from controller import PayloadController
from ipc_comms import PayloadIPC
from definitions import PayloadTM, Resp_EnableCameras, Resp_DisableCameras


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

        if resp:
            counter += 1
            print("[INFO] Ping succeeded.")
        else:
            print("[ERROR] Ping failed.")

        time.sleep(0.5)
        print(f"[INFO] {counter}/{i} ping(s) succeeded.")

    # Testing telemetry request
    print("[INFO] Requesting telemetry...")
    controller.request_telemetry()

    resp = controller.receive_response()
    print(resp)
    if resp:
        print("[INFO] Telemetry received.")
        PayloadTM.print()

    # Testing camera disable and renable 
    print("[INFO] Disabling cameras...")
    controller.disable_cameras()
    time.sleep(0.2)
    resp = controller.receive_response()
    if resp:
        print(f"[INFO] {Resp_DisableCameras.num_deactivated_cameras} cameras disabled.")

    time.sleep(1) 

    print("[INFO] Enabling cameras...")
    controller.enable_cameras()
    time.sleep(0.2)
    resp = controller.receive_response()
    if resp:
        print(f"[INFO] {Resp_EnableCameras.num_activated_cameras} cameras enabled.")

    time.sleep(1)

    controller.shutdown()

    print("[INFO] Done.")
    controller.deinitialize()
    exit(0)