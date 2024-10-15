import socket
import pickle
from ultralytics import YOLO
from PIL import Image
import io

# Constants
CLASS_NAMES = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle"}

# Object Detection
def detect_object(image_data, class_id, model_path='yolov8n.pt'):
    model = YOLO(model_path)
    image = Image.open(io.BytesIO(image_data))
    results = model(image)
    # Filter detections by class ID
    object_detections = [box for result in results for box in result.boxes if int(box.cls) == class_id]
    return object_detections

# Data Reception
def receive_image_data(client_socket):
    length_bytes = client_socket.recv(8)
    data_length = int.from_bytes(length_bytes, byteorder='big')
    print(f"Client expecting {data_length} bytes of data")
    data = b""
    while len(data) < data_length:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data += packet

    print(f"Client received {len(data)} bytes")
    
    return pickle.loads(data)

# Client Communication
def start_client(object_type):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('192.168.1.11', 8095))
    print(f"Client for {CLASS_NAMES[object_type]} connected to server")
    
    # Send object type to the server
    client_socket.sendall(object_type.to_bytes(4, byteorder="big"))
    
    # Receive image data from the server
    image_data = receive_image_data(client_socket)
    print(f"Client for {CLASS_NAMES[object_type]} received image data")
    
    # Detect objects of the assigned type
    object_detections = detect_object(image_data, object_type)
    print(f"Client for {CLASS_NAMES[object_type]} detected object")
    
    # Send detection results back to the server
    client_socket.sendall(pickle.dumps(object_detections))
    print(f"Sent {CLASS_NAMES[object_type]} detection results back to server")
    client_socket.close()

#================== Example Usage
if __name__ == "__main__":
    object_type = 2  # This client detects Car
    start_client(object_type)
