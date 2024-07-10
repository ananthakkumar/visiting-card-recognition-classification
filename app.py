from flask import Flask, request, jsonify, render_template
import cv2
import pytesseract
from langchain_community.llms import Ollama

# Initialize the Ollama client with the correct model
llm = Ollama(model="llama2")

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def extract_text(image_path):
  image = cv2.imread(image_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
  return pytesseract.image_to_string(binary)

def generate_structured_information(text):
  prompt = (
        f"Extract and organize the following text into structured information including "
        f"company name, service type, name, position, phone number, email address, website, and address:\n\n"
        f"{text}\n\n"
        f"Provide the information in the following JSON format:\n"
        f"{{\n"
        f"  \"Company Information\": {{\n"
        f"    \"Company Name\": \"\",\n"
        f"    \"Service Type\": \"\"\n"
        f"  }},\n"
        f"  \"Personal Information\": {{\n"
        f"    \"Name\": \"\",\n"
        f"    \"Position\": \"\"\n"
        f"  }},\n"
        f"  \"Contact Information\": {{\n"
        f"    \"Phone Number\": \"\",\n"
        f"    \"Email Address\": \"\",\n"
        f"    \"Website\": \"\"\n"
        f"  }},\n"
        f"  \"Address\": {{\n"
        f"    \"Location\": \"\"\n"
        f"  }}\n"
        f"}}"
    )
  
  response = llm(prompt)
  return response

@app.route('/')
def upload_form():
  return render_template('index.html')

@app.route('/process_card', methods=['POST'])
def process_card():
  if 'file' not in request.files:
    return jsonify({"error": "No file provided"}), 400

  file = request.files['file']
  if file.filename == '':
    return jsonify({"error": "No selected file"}), 400

  file_path = f"temp_{file.filename}"
  file.save(file_path)

  # Extract text from the image
  text = extract_text(file_path)

  # Generate structured information from the text
  structured_info = generate_structured_information(text)

  return jsonify(structured_info)

if __name__ == '__main__':
  app.run(debug=True)
