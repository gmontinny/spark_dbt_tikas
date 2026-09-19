import requests, sys

r = requests.put(
    "http://tika-server:9998/tika",
    data=open("/opt/spark/work-dir/datas/ipop-dezembro-2025.pdf", "rb").read(),
    headers={"Accept": "text/plain", "Content-Type": "application/pdf"},
)
sys.stdout.write("STATUS: " + str(r.status_code) + "\n")
sys.stdout.write("LEN: " + str(len(r.text)) + "\n")
sys.stdout.write("PREVIEW: " + r.text[:500] + "\n")

# Testa também com OCR header
r2 = requests.put(
    "http://tika-server:9998/tika",
    data=open("/opt/spark/work-dir/datas/ipop-dezembro-2025.pdf", "rb").read(),
    headers={
        "Accept": "text/plain",
        "Content-Type": "application/pdf",
        "X-Tika-OCRLanguage": "por+eng",
        "X-Tika-PDFextractInlineImages": "true",
    },
)
sys.stdout.write("OCR STATUS: " + str(r2.status_code) + "\n")
sys.stdout.write("OCR LEN: " + str(len(r2.text)) + "\n")
sys.stdout.write("OCR PREVIEW: " + r2.text[:500] + "\n")
