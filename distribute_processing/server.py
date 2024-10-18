import socket
import pickle
from threading import Thread
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Constants
CLASS_NAMES = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle"}
COLOR_MAP = {
    0: (255, 0, 0),      # Red for Person
    1: (0, 255, 255),    # Cyan for Bicycle
    2: (0, 255, 0),      # Green for Car
    3: (0, 0, 255),      # Blue for Motorcycle
}

# Image Communication Functions
def send_image_to_client(client_socket, image_data):
    """Send image data to the client, starting with its length."""
    image_data_pickle = pickle.dumps(image_data)
    data_length = len(image_data_pickle)
    client_socket.sendall(data_length.to_bytes(8, byteorder="big")) 
    client_socket.sendall(image_data_pickle) 

def receive_detection_from_client(client_socket):
    """Receive object detection results from a client."""
    data = b""
    while True:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data += packet
    return pickle.loads(data)

def receive_object_type_from_client(client_socket):
    """Receive the object type that the client will detect."""
    object_type_data = client_socket.recv(4) 
    object_type = int.from_bytes(object_type_data, byteorder='big')
    return object_type

# Client Handling
def handle_client(client_socket, addr, image_data, results):
    """Handle a single client connection and process detection."""
    object_type = receive_object_type_from_client(client_socket)
    print(f"Connected to client {addr}, detecting: {CLASS_NAMES[object_type]}")
    send_image_to_client(client_socket, image_data)
    client_results = receive_detection_from_client(client_socket)
    results[object_type] = client_results
    print(f"Received {CLASS_NAMES[object_type]} results from client {addr}")
    client_socket.close()

# Server Initialization
def start_server(image_path):
    """Initialize the server, manage client connections, and aggregate results."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("10.26.14.197", 8095))  
    server_socket.listen(4)  # Listening for 4 clients

    print("Server is waiting for clients to connect...")

    # Load the image data
    with open(image_path, "rb") as f:
        image_data = f.read()

    results = {}  

    # Accept connections from 4 clients
    connected_clients = 0  
    while connected_clients < 4:
        client_socket, addr = server_socket.accept()
        connected_clients += 1 
        client_thread = Thread(
            target=handle_client,
            args=(client_socket, addr, image_data, results)
        )
        client_thread.start()

    # Wait for all threads to finish
    for _ in range(connected_clients):
        client_thread.join()

    server_socket.close()
    print("Final aggregated results:", results)
    display_image_with_detections(image_path, results, "detected_objects_image.jpg")
    return results

# Image Processing and Visualization
def display_image_with_detections(image_path, results, output_path):
    """Display the image with object detections and save it to a file."""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Iterate through each object's detections and draw bounding boxes
    for object_type, detections in results.items():
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  
            confidence = detection.conf[0].item()  
            color = COLOR_MAP.get(object_type)  
            # Draw bounding box and label on the image
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1), f"{CLASS_NAMES[object_type]}: {confidence:.2f}", fill=color)

    image.save(output_path)
    print(f"Image saved to {output_path}")


#================== Example Usage
if __name__ == "__main__":
    image_path = "image.jpg"  
    aggregated_results = start_server(image_path)
