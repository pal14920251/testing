#pdf text extraction
import pdfplumber
import fitz
from PIL import Image
import pytesseract

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
3. Use precise agricultural and scientific terminology.
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

from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json

client = OpenAI()

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
def generate_qa(text):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}
        ],
        temperature=0.4
    )

    content = response.choices[0].message.content
    return json.loads(content)

#Store in Mistral-Instruct Format

def save_qa_pairs(qa_data, output_file):
    with open(output_file, "a", encoding="utf-8") as f:
        for pair in qa_data["qa_pairs"]:
            record = {
                "instruction": "Answer the following agriculture-related question.",
                "input": pair["question"],
                "output": pair["answer"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
#Main Loop (Streaming PDFs One by One)
import os
from tqdm import tqdm

PDF_ROOT = "C:\tester"
OUTPUT_FILE = "C:\tester\qa_dataset.jsonl"

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
                    qa_data = generate_qa(chunk)
                    save_qa_pairs(qa_data, OUTPUT_FILE)
                except Exception as e:
                    print(f"Failed chunk in {file}: {e}")
