from langchain_community.llms import Ollama
import cv2
import pytesseract
import json

# Initialize the Ollama client with the correct model
llm = Ollama(model="llama2")

# Extract text from the image
def extract_text(image_path):
    return pytesseract.image_to_string(image_path)

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
    
    print("Raw response:", response) 
    
    try:
        response_json = json.loads(response)
        return response_json
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None

def save_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
img_path ="C:/002JesperApps internship/001Visiting card recognition/img/card1.png"

txt = extract_text(img_path)
res = generate_structured_information(txt)


import pickle
with open('model.pkl','wb') as f:
    pickle.dump(llm,f)