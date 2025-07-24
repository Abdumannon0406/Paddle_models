from paddleocr import FormulaRecognition
model = FormulaRecognition(model_name="PP-FormulaNet_plus-M")
output = model.predict(input="formula2.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="output/")
    res.save_to_json(save_path="output/formula.json")
