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
    outputs = model.generate(
        **inputs,
        max_length=200,  # Increased length
        temperature=0.9,  # Adds randomness
        top_p=0.9,  # Limits word selection to high-probability words
        repetition_penalty=1.2,  # Avoids repeating the same phrase
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
