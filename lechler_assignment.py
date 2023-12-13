import cv2
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz

#Locally downloading subset of COCO subset using Fiftyone

dataset=foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["segmentations"],
    classes=["cat", "dog"],
    max_samples=25,
)

print("DATASET_INFO")
print(dataset)


image_path='/home/susmit/fiftyone/coco-2017/validation/data/000000022192.jpg'
image=cv2.imread(image_path)

## Apply Gaussian Blur filter
kernel=(3,3)
sigma=1.5
image=cv2.GaussianBlur(image, kernel, sigma)

## Load pretrained yolov3 dataset using yolo.cfg and yolo.weights(pre trained weights)

net=cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes=None
with open("yolov3.txt", "r") as f:
    classes=[line.strip() for line in f.readlines()]

layer_names=net.getLayerNames()
output_layers_indices=net.getUnconnectedOutLayers()
output_layers=[layer_names[i - 1] for i in output_layers_indices]


height, width, channels=image.shape
blob=cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
output=net.forward(output_layers)


class_ids = []
confidences = []
boxes = []

for out in output:
    for detection in out:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]
        if confidence>0.5:
            center_x=int(detection[0]*width)
            center_y=int(detection[1]*height)
            w=int(detection[2]*width)
            h=int(detection[3]*height)
            x=int(center_x-w/2)
            y=int(center_y-h/2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices=cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
for i in indices:
    box=boxes[i]
    x, y, w, h=box
    label=classes[class_ids[i]]
    confidence=confidences[i]
    color=(0, 255, 0)
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

