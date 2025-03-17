"""
Low-Level Communication layer - Named Pipe (FIFO)

This is a perfect replica of the C++ CLI command line process implemented in src/cli_cmd.cpp.

Author: Ibrahima Sory Sow
"""

import os
import sys
import threading
import readline
import select

# Named pipe paths (make sure this corresponds to the CMAKE compile definitions)
FIFO_IN = "/tmp/payload_fifo_in"  # Payload reads from this, external process writes to it
FIFO_OUT = "/tmp/payload_fifo_out"  # Payload writes to this, external process reads from it

RUNNING = True  # Global flag for stopping the response reader


def is_fifo(path):
    """Check if a path is a named pipe (FIFO)."""
    return os.path.exists(path) and os.path.stat(path).st_mode & 0o170000 == 0o10000


def read_responses(fifo_out):
    """Background thread function to read responses from FIFO without blocking."""
    global RUNNING
    try:
        # Open the FIFO for reading (non-blocking mode)
        fd = os.open(fifo_out, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        print(f"Error opening FIFO for reading: {e}")
        return

    while RUNNING:
        rlist, _, _ = select.select([fd], [], [], 2)  # Wait up to 2s for data
        if fd in rlist:
            try:
                data = os.read(fd, 512).decode().strip()
                if data:
                    # Clear current input and print the message
                    sys.stdout.write("\r\033[K")  # Clear line
                    print(f"\033[1;31m[PAYLOAD RESPONSE]:\033[0m {data}")
                    sys.stdout.write("PAYLOAD> " + readline.get_line_buffer())  # Restore input
                    sys.stdout.flush()
            except OSError:
                pass

    os.close(fd)


def main():
    global RUNNING

    # Ensure the FIFOs exist
    for fifo in [FIFO_IN, FIFO_OUT]:
        if not os.path.exists(fifo):
            try:
                os.mkfifo(fifo, 0o666)
            except OSError as e:
                print(f"Error creating FIFO {fifo}: {e}")
                return 1

    # Open the input FIFO for writing
    try:
        pipe_out = open(FIFO_IN, "w", buffering=1)  # Line-buffered
    except OSError as e:
        print(f"Error: Could not open FIFO {FIFO_IN} for writing: {e}")
        return 1

    # Start background response reader thread
    response_reader_thread = threading.Thread(target=read_responses, args=(FIFO_OUT,), daemon=True)
    response_reader_thread.start()

    # Command input loop
    try:
        while True:
            cmd = input("PAYLOAD> ").strip()
            if cmd:
                readline.add_history(cmd)
                pipe_out.write(cmd + "\n")
                pipe_out.flush()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")

    RUNNING = False
    response_reader_thread.join()
    pipe_out.close()


if __name__ == "__main__":
    sys.exit(main())
