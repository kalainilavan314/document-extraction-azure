# Intelligent Document Extraction Pipeline (In Progress)
This project demonstrates an end-to-end **document extraction and OCR pipeline** built using **Microsoft Azure AI services** and Python.  
It focuses on extracting text accurately from **blurry, unclear, and mixed-format documents** stored in cloud storage.

The project is currently **under active development** and is being enhanced step by step with advanced AI techniques.

---

## 🔍 Problem Statement
Real-world documents are often:
- Blurry or low resolution
- Scanned or photographed
- In mixed formats (PDF, PNG, JPG, TXT, XLSX)
- Poorly structured for traditional OCR

Traditional OCR solutions struggle with these conditions.  
This project explores **cloud-native, AI-driven approaches** to improve accuracy and reliability.

## 🏗️ Architecture Overview
Azure Blob Storage
↓
Document Download (SAS-secured)
↓
Image Preprocessing (OpenCV)
↓
Azure AI Document Intelligence (OCR)
↓
Clean Text Output

## 🧠 Key Features (So Far)
- Secure document access using **Azure Blob Storage + SAS**
- Image preprocessing for unclear documents:
  - Upscaling
  - Denoising
  - Sharpening
  - Thresholding
- OCR using **Azure AI Document Intelligence (Read model)**
- Supports multiple document types:
  - PDFs
  - Images (PNG / JPG)
  - Text files
- Clean and readable text extraction
- Designed for scalability and cloud deployment

## 🛠️ Technologies Used
- Python 3
- Azure Blob Storage
- Azure AI Document Intelligence
- OpenCV
- Pillow
- LangChain (Azure Storage integration)
- python-dotenv

## 📂 Project Structure
Document_Extraction/
├── main.py # Core extraction pipeline
├── .gitignore # Git ignore rules
└── README.md # Project documentation

## ▶️ How to Run (Local)
1. Clone the repository:
   ```bash
   git clone https://github.com/kalainilavan314/document-extraction-azure.git
   cd document-extraction-azure
Create and activate virtual environment:

python3 -m venv .venv
source .venv/bin/activate
Install dependencies:

pip install -r requirements.txt
Configure environment variables:

Create a .env file (not committed)

Add Azure Blob and Azure AI credentials

Planned enhancements:
Document classification
Structured field extraction
Confidence scoring
Validation and error handling
AI-powered querying and retrieval
Output storage (JSON / database)

📌 Key Learnings
OCR accuracy improves significantly with proper preprocessing
Cloud-based AI services outperform traditional OCR on unclear documents
Secure access and scalability are essential for real-world systems
Document AI is more than just text extraction — structure and confidence matter

📜 Disclaimer
This project is for learning and experimentation purposes and is not yet production-ready.

👤 Author
Kalainilavan Suthan
AI / Digital Transformation Engineer
