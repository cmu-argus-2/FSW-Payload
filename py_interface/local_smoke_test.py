import time 
from controller import PayloadController, PayloadState
from ipc_comms import PayloadIPC
from definitions import PayloadTM, Resp_EnableCameras, Resp_DisableCameras, FileTransfer, FileTransferType


if __name__ == '__main__':

    TIMEOUT = 1 # seconds

    ipc = PayloadIPC
    controller = PayloadController
    controller.inject_communication_interface(ipc)
    controller.initialize()

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

        time.sleep(0.2)
        print(f"[INFO] {counter}/{i} ping(s) succeeded.")


    # SPOOF the payload as READY for now
    controller.state = PayloadState.READY


    # Testing telemetry request
    print("[INFO] Requesting telemetry...")
    controller.request_telemetry()

    time.sleep(0.1)
    resp = controller.receive_response()
    print(resp)
    if resp:
        print("[INFO] Telemetry received.")
        PayloadTM.print()

    # Testing camera disable and renable 
    print("[INFO] Disabling cameras...")
    controller.disable_cameras()
    time.sleep(0.5)
    resp = controller.receive_response()
    if resp:
        print(f"[INFO] {Resp_DisableCameras.num_cam_deactivated} cameras disabled.")

    time.sleep(1) 

    print("[INFO] Enabling cameras...")
    controller.enable_cameras()
    time.sleep(0.5) # Takes time to enable the cameras
    resp = controller.receive_response()
    if resp:
        print(f"[INFO] {Resp_EnableCameras.num_cam_activated} cameras enabled.")

    time.sleep(1)

    ## Testing image transfer
    print("[INFO] Requesting image transfer...")
    controller.request_image_transfer()

    time.sleep(0.5)
    resp = controller.receive_response()
    print(resp)
    if resp:
        print("[INFO] Image transfer started.")
        FileTransfer.start_transfer(FileTransferType.IMAGE)

        MAX_PACKETS = 25000

        while FileTransfer.packet_nb < MAX_PACKETS:
            
            res = controller._continue_file_transfer_logic()

            if res:
                print(f"[INFO] {FileTransfer.packet_nb} packets received.")

            if controller.no_more_file_packet_to_receive:
                print("[INFO] No more packets to receive. We're done here.")
                break

            # to skip packets (avoid waiting)
            if FileTransfer.packet_nb == 5:
                FileTransfer.packet_nb = (93 << 8) | 240

    else:
        print("[ERROR] Image transfer request failed.")


    #controller.shutdown()

    print("[INFO] Done.")
    controller.deinitialize()
    exit(0)