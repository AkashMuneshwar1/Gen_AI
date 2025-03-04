from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_DIR = "trained_model"

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Validate if the response is relevant
    if prompt.lower() not in generated_text.lower():
        return jsonify({"generated_text": "❌ Not valid. The query is not in the trained model."})

    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
