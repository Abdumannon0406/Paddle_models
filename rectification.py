from paddleocr import TextImageUnwarping
model = TextImageUnwarping(model_name="UVDoc")
output = model.predict("book.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="output/")
    res.save_to_json(save_path="output/book_res.json")