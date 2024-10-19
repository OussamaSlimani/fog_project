import socket
import pickle
from ultralytics import YOLO
from PIL import Image
import io

# Constants
CLASS_NAMES = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle"}

# Object Detection
def detect_object(image_data, class_ids, model_path='yolov8n.pt'):
    model = YOLO(model_path)
    
    # Ensure the image data is valid and can be opened
    try:
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    
    results = model(image)
    
    # Filter detections by class IDs
    detections = {cls_id: [] for cls_id in class_ids}
    for result in results:
        for box in result.boxes:
            if int(box.cls) in class_ids:
                detections[int(box.cls)].append(box)
    return detections

def receive_data(client_socket, expected_size):
    data = b""
    while len(data) < expected_size:
        packet = client_socket.recv(4096)
        if not packet:
            raise ConnectionError("Connection lost while receiving data.")
        data += packet
    return data

def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('192.168.1.12', 8095))
    print("Client connected to server")

    try:
        while True:
            message = client_socket.recv(1024)
            if not message:
                print("No message received. Closing connection.")
                break

            try:
                message = pickle.loads(message)
            except pickle.UnpicklingError:
                print("Failed to unpickle the received message. Invalid format.")
                break

            if message == "Check availability":
                response = input("Are you available? (yes/no): ").strip().lower()
                client_socket.sendall(response.encode("utf-8"))
                if response == "no":
                    break
                else:
                    # Receive the length of the assigned objects data
                    length_bytes = receive_data(client_socket, 8)
                    assigned_objects_length = int.from_bytes(length_bytes, byteorder='big')

                    # Receive the actual assigned objects data
                    assigned_objects_data = receive_data(client_socket, assigned_objects_length)
                    assigned_objects = pickle.loads(assigned_objects_data)
                    print(f"Assigned to detect objects: {assigned_objects}")

                    # Receive the image data length
                    image_data_length_bytes = receive_data(client_socket, 8)
                    image_data_length = int.from_bytes(image_data_length_bytes, byteorder="big")

                    # Receive the actual image data
                    image_data = receive_data(client_socket, image_data_length)

                    # Check if image_data is empty or malformed
                    if not image_data:
                        print("Received empty image data.")
                        break

                    # Detect objects of the assigned type(s)
                    object_detections = detect_object(image_data, assigned_objects)

                    # Send detection results back to the server
                    client_socket.sendall(pickle.dumps(object_detections))
                    print("Sent detection results back to server")

                    break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client_socket.close()

#================== Example Usage
if __name__ == "__main__":
    start_client()
