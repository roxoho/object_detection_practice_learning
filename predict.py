"""import os
from ultralytics import YOLO
import cv2

IMAGES_DIR = os.path.join('.', 'videos')

image_path = os.path.join(IMAGES_DIR, 'image.jpg')
output_path = '{}_out.jpg'.format(image_path)

# Read the image
image = cv2.imread(image_path)
H, W, _ = image.shape

model_path = os.path.join('.', 'runs', 'detect', 'train5', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0

# Convert image to BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform prediction
results = model(image)[0]
print(results)

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Convert back to RGB
#image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Save the output image
cv2.imwrite(output_path, image)

# Display the original and output images (optional)
cv2.imshow('Original Image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imshow('Output Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
