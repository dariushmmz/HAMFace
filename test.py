from ultralytics import YOLO
import cv2
import os

# Load model
model = YOLO("checkpoints/yolov12n-face.pt")

image = cv2.imread("inputs/dariush.JPG")
# H, W, _ = image.shape
# image = cv2.resize(image, (W//2, H//2))
results = model(image, conf=0.3)

os.makedirs("faces", exist_ok=True)

for result in results:

    boxes = result.boxes.xyxy.cpu().numpy()

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = map(int, box)

        # Draw rectangle
        cv2.rectangle(image,
                      (x1, y1),
                      (x2, y2),
                      (0,255,0),
                      2)

        # Crop face
        face = image[y1:y2, x1:x2]

        cv2.imwrite(f"faces/face_{i}.jpg", face)

# cv2.imwrite("output.jpg", image)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()