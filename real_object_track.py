import cv2
from ultralytics import YOLO

def initialize_yolo_model(model_path):
    return YOLO(model_path)

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height))

def select_object(event, x, y, flags, param):
    global selected_object_coords

    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse Clicked at:", x, y)
        selected_object_coords = (x, y)

def track_objects(cap, model, frame_width, frame_height):
    selected_object_coords = None
    cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("YOLOv8 Tracking", select_object)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Resize the frame
            frame = resize_frame(frame, frame_width, frame_height)
            
            results = model.track(frame, persist=True)

            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

def main(video_path, model_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2  # Resize width
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2  # Resize height
    model = initialize_yolo_model(model_path)
    track_objects(cap, model, frame_width, frame_height)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "Test.mp4"
    model_path = 'yolov9c.pt'
    main(video_path, model_path)
