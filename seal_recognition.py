from paddleocr import SealTextDetection
model = SealTextDetection(model_name="PP-OCRv4_server_seal_det")
output = model.predict("seal_image.png", batch_size=1)

for res in output:
    res.print()
    res.save_to_img(save_path="output/")
    res.save_to_json(save_path="output/seal.json")