# Socket program for the deployment 

import socket as socket_lib
import struct
from threading import Thread
from time import sleep
from queue import Queue
from typing import Tuple
import requests

socket_queue: Queue[Tuple[str, bool]] = Queue()


def __start_socket() -> None:
    while True:
        try:
            socket = socket_lib.socket(socket_lib.AF_INET, socket_lib.SOCK_STREAM)
            print("Socket: Connecting to localhost:4999")
            socket.connect(("localhost", 4999))
            print("Socket: Connected!")

            while True:
                msg, is_stt = socket_queue.get()
                if is_stt:
                    socket.send(bytes([1]))
                else:
                    socket.send(bytes([0]))
                payload = msg.encode("utf-8")
                payload_length: bytes = struct.pack("!i", len(payload))
                socket.sendall(payload_length)
                socket.sendall(payload)
        except Exception as e:
            print(f"Socket: Disconnected due to the following error: {e}")
            sleep(0.3)
            continue


def SendMessage(msg: str, is_stt: bool) -> None:
    if is_stt:
        requests.post("http://localhost:5555/message", data=msg.encode("utf-8"))
    else:
        socket_queue.put_nowait((msg, False))


Thread(target=__start_socket).start()
