from paddleocr import TextDetection
model = TextDetection(model_name="PP-OCRv5_server_det")
output = model.predict("massage.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="output/")
    res.save_to_json(save_path="output/text_detection_res.json")