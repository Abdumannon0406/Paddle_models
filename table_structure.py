from paddleocr import TableStructureRecognition
import numpy as np
import cv2

image_path="table.png"
model = TableStructureRecognition(model_name="SLANeXt_wireless")
output = model.predict(input=image_path, batch_size=1)
# print(type(output))
for res in output:
    res.print(json_format=False)
    res.save_to_json("output/table_structure_res.json")


image=cv2.imread(image_path)


height,width,channel=image.shape

new_image=np.ones((height,width))*255

boxes=output[0]["bbox"]
for i in range(len(boxes)):
    pts = np.array(boxes[i], dtype=np.int32).reshape((-1, 1, 2))
    rectange=cv2.polylines(new_image,[pts],isClosed=True,color=(0, 0, 0),thickness=1)

cv2.imwrite("output.png",rectange)

