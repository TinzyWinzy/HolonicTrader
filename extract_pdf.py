from pypdf import PdfReader

pdf_path = "Academic_White_Paper_on_AEHML_Framework-1.pdf"
output_path = "white_paper_full.txt"

try:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"Full text saved to {output_path}")

except Exception as e:
    print(f"Error: {e}")
