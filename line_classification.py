from paddleocr import TextLineOrientationClassification
model = TextLineOrientationClassification(model_name="PP-LCNet_x0_25_textline_ori")
output = model.predict("2247238__integra.jpg",  batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("output/")
    res.save_to_json("output/line_orientation_res.json")