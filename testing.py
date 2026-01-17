#pdf text extraction
import pdfplumber
import fitz
from PIL import Image
import pytesseract



# ================================
import vertexai
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, Part

class GeminiAssistant:
    def __init__(self, project_id, key_path, location="asia-south1"):
        # Initialize credentials and Vertex AI
        self.credentials = service_account.Credentials.from_service_account_file(key_path)
        vertexai.init(project=project_id, location=location, credentials=self.credentials)
        
    def get_response(self, system_instruction, user_query, model_name="gemini-2.5-flash"):
        # Initialize the model WITH the system instructions
        model = GenerativeModel(
            model_name=model_name,
            system_instruction=[system_instruction]
        )
        
        # Generate the response using the user prompt
        response = model.generate_content(user_query)
        return response.text

# --- USAGE ---

# 1. Configuration
PROJECT_ID = "genai-angelone"
KEY_FILE = "genai-angelone.json"

# 2. Instantiate your systematic client
ai = GeminiAssistant(PROJECT_ID, KEY_FILE)

# 3. Define your Systematic Prompts
my_system_prompt = """
You are a Cloud Security Expert specializing in AWS S3 and Databricks. 
Your goal is to help users identify PII (Personally Identifiable Information).
Always explain WHY a specific ID (like BEN_ID) is considered PII.
Be concise and professional.
"""

my_user_query = "Is a column named 'FOLIO_NUMBER' considered PII in a mutual fund dataset?"

# 4. Get the result
# result = ai.get_response(my_system_prompt, my_user_query)
# print(result)
# ================================

def extract_text_from_pdf(pdf_path):
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 200:
                pages.append(text)

    # OCR fallback
    if not pages:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)
            if len(ocr_text.strip()) > 200:
                pages.append(ocr_text)

    return pages

#Chunking (Context Control)
def chunk_text(text, max_words=600):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.split()) > 200:
            chunks.append(chunk)

    return chunks

#Q&A Generation Prompt

SYSTEM_PROMPT = """
You are an expert agriculture researcher and university professor.
"""

USER_PROMPT_TEMPLATE = """
Given the following paragraph from an academic agriculture text:

1. Generate 3–5 high-quality questions that test conceptual understanding.
2. Answer each question in a descriptive, textbook-style manner.
3. Use accurate agricultural and scientific terminology,but explain ideas in simple,clear language so that an educated non-expert can understand them,while still preserving all essential technical and scientific meaning.
4. Do NOT reference the paragraph explicitly.
5. Output strictly valid JSON.

Paragraph:
<<<{text}>>>

Output format:
{{
  "qa_pairs": [
    {{
      "question": "...",
      "answer": "..."
    }}
  ]
}}
"""
#Call Large LLM (Example: OpenAI-style)

# from openai import OpenAI
# from tenacity import retry, wait_random_exponential, stop_after_attempt
import json

# client = OpenAI()
count=0
# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
def generate_qa(text):
 
    result = ai.get_response(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(text=text))
    # response = client.chat.completions.create(
    #     model="gpt-4.1",
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}
    #     ],
    #     temperature=0.4
    # )

    # content = response.choices[0].message.content
    return result

#Store in Mistral-Instruct Format

def save_qa_pairs(qa_data, output_file):
    print("="*25)
    print(qa_data)
    pass
    # with open(output_file, "a", encoding="utf-8") as f:
    #     for pair in qa_data["qa_pairs"]:
    #         record = {
    #             "instruction": "Answer the following agriculture-related question.",
    #             "input": pair["question"],
    #             "output": pair["answer"]
    #         }
    #         f.write(json.dumps(record, ensure_ascii=False) + "\n")
#Main Loop (Streaming PDFs One by One)
import os
from tqdm import tqdm

PDF_ROOT = "./"
OUTPUT_FILE = "qa_dataset.jsonl"
count=0
for root, _, files in os.walk(PDF_ROOT):
    for file in tqdm(files):
        if not file.endswith(".pdf"):
            continue

        pdf_path = os.path.join(root, file)
        pages = extract_text_from_pdf(pdf_path)

        for page_text in pages:
            chunks = chunk_text(page_text)

            for chunk in chunks:
                try:
                    count+=1
                    qa_data = generate_qa(chunk)
                    save_qa_pairs(qa_data, OUTPUT_FILE)
                except Exception as e:
                    print(f"Failed chunk in {file}: {e}")
print("Processing complete.")
print(f"total API calls made: {count}")