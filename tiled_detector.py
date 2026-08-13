import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8s.pt")  # or yolov8n.pt if RAM is low

# -----------------------------
# Function: Split image into tiles
# -----------------------------
def tile_image(image, tiles_x=2, tiles_y=2):
    h, w, _ = image.shape
    tile_w = w // tiles_x
    tile_h = h // tiles_y

    tiles = []
    positions = []

    for i in range(tiles_y):
        for j in range(tiles_x):
            x1 = j * tile_w
            y1 = i * tile_h
            x2 = x1 + tile_w
            y2 = y1 + tile_h

            tile = image[y1:y2, x1:x2]
            tiles.append(tile)
            positions.append((x1, y1))

    return tiles, positions

# -----------------------------
# Function: Detect using tiling
# -----------------------------
def detect_with_tiling(image):
    tiles, positions = tile_image(image, 2, 2)
    all_boxes = []

    for tile, (x_offset, y_offset) in zip(tiles, positions):
        results = model(tile, conf=0.15)

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # person
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    x1 += x_offset
                    x2 += x_offset
                    y1 += y_offset
                    y2 += y_offset

                    all_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    return all_boxes

# -----------------------------
# Function: Generate Heatmap
# -----------------------------
def generate_heatmap(image, boxes):

    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    for (x1, y1, x2, y2) in boxes:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.circle(heatmap, (cx, cy), 50, 1, -1)

    heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)

    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(
        heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET
    )

    output = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

    return output

# -----------------------------
# Process all frames in folder
# -----------------------------
input_folder = "data/frames/all"
output_folder = "data/frames/detected"

os.makedirs(output_folder, exist_ok=True)

total_frames = 0

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        boxes = detect_with_tiling(image)

        # Draw bounding boxes
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 🔥 APPLY HEATMAP
        output = generate_heatmap(image, boxes)

        # Save final image
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, output)

        print(f"{filename} → Persons detected: {len(boxes)}")

        total_frames += 1

print("\nFinished Processing!")
print("Total frames processed:", total_frames)