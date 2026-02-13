import os
import io
import logging
import tempfile
from dotenv import load_dotenv

import cv2
import numpy as np
from PIL import Image

from azure.core.credentials import AzureKeyCredential, AzureSasCredential
from azure.storage.blob import BlobClient

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

# Reduce noisy logs
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

load_dotenv()

# ----------------------------
# ENV: Azure Blob (SAS)
# ----------------------------
ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")  # e.g. https://blobtestaccount01.blob.core.windows.net
CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")      # e.g. municeabilene
SAS = os.getenv("AZURE_STORAGE_SAS_TOKEN")            # SAS token only
BLOB_NAME = os.getenv("AZURE_STORAGE_SINGLE_BLOB")    # e.g. 1/Screenshot (2357).png

if not all([ACCOUNT_URL, CONTAINER, SAS, BLOB_NAME]):
    raise ValueError("Missing one of: AZURE_STORAGE_ACCOUNT_URL, AZURE_STORAGE_CONTAINER, AZURE_STORAGE_SAS_TOKEN, AZURE_STORAGE_SINGLE_BLOB")

# ----------------------------
# ENV: Azure Document Intelligence
# ----------------------------
DOC_ENDPOINT = os.getenv("AZURE_DOCINTEL_ENDPOINT")
DOC_KEY = os.getenv("AZURE_DOCINTEL_KEY")
MODEL_ID = os.getenv("AZURE_DOCINTEL_MODEL", "prebuilt-read")

if not all([DOC_ENDPOINT, DOC_KEY]):
    raise ValueError("Missing AZURE_DOCINTEL_ENDPOINT or AZURE_DOCINTEL_KEY in .env")

# ----------------------------
# 1) Download blob bytes
# ----------------------------
def download_blob_bytes() -> bytes:
    cred = AzureSasCredential(SAS.lstrip("?"))
    blob = BlobClient(
        account_url=ACCOUNT_URL.rstrip("/"),
        container_name=CONTAINER,
        blob_name=BLOB_NAME,
        credential=cred,
    )
    data = blob.download_blob().readall()
    return data

# ----------------------------
# 2) Preprocess image (blur/unclear)
#    - upscale
#    - denoise
#    - sharpen
#    - adaptive threshold (optional)
# ----------------------------
def preprocess_image_to_png_bytes(img_bytes: bytes) -> bytes:
    # Load with PIL then convert to OpenCV BGR
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Upscale 2x (helps small/blur text)
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    den = cv2.fastNlMeansDenoising(gray, h=12)

    # Sharpen (unsharp mask)
    blur = cv2.GaussianBlur(den, (0, 0), sigmaX=1.2)
    sharp = cv2.addWeighted(den, 1.7, blur, -0.7, 0)

    # Adaptive threshold to make text clearer (good for scanned docs)
    thr = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    # Encode to PNG bytes
    ok, buf = cv2.imencode(".png", thr)
    if not ok:
        raise RuntimeError("Failed to encode preprocessed image to PNG")
    return buf.tobytes()

# ----------------------------
# 3) Send to Azure Document Intelligence (Read model)
# ----------------------------
def extract_text_with_doc_intel(file_bytes: bytes) -> str:
    client = DocumentIntelligenceClient(
        endpoint=DOC_ENDPOINT,
        credential=AzureKeyCredential(DOC_KEY)
    )

    poller = client.begin_analyze_document(
        model_id=MODEL_ID,  # "prebuilt-read" for full OCR text
        body=file_bytes,
        content_type="application/octet-stream",
    )
    result = poller.result()

    # Collect lines in reading order
    lines = []
    if result.pages:
        for page in result.pages:
            if page.lines:
                for line in page.lines:
                    if line.content:
                        lines.append(line.content)

    return "\n".join(lines).strip()

def main():
    print("Downloading blob:", BLOB_NAME)
    raw = download_blob_bytes()
    print("Downloaded bytes:", len(raw))

    print("Preprocessing image for better OCR...")
    cleaned = preprocess_image_to_png_bytes(raw)
    print("Preprocessed bytes:", len(cleaned))

    print("Running Azure Document Intelligence OCR (model:", MODEL_ID, ") ...")
    text = extract_text_with_doc_intel(cleaned)

    print("\n===== FULL EXTRACTED TEXT =====\n")
    print(text)
    print("\n===== END =====\n")
    print("Total characters:", len(text))

if __name__ == "__main__":
    main()
