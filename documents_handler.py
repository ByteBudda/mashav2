# document_handler.py
import os
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import io
import logging

# Configure logger
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_pdf(file_path):
    """Чтение текста из PDF файла."""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        logger.error(f"Ошибка чтения PDF: {e}")
        return None

def read_docx(file_path):
    """Чтение текста из DOCX файла."""
    try:
        doc = DocxDocument(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Ошибка чтения DOCX: {e}")
        return None

def read_txt(file_path):
    """Чтение текста из TXT файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Ошибка чтения TXT: {e}")
        return None

def read_py(file_path):
    """Чтение текста из PY файла."""
    return read_txt(file_path)

def generate_document(content, file_path, file_type='txt'):
    """Генерация документа заданного типа."""
    try:
        if file_type == 'txt':
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
        elif file_type == 'docx':
            doc = DocxDocument()
            doc.add_paragraph(content)
            doc.save(file_path)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        logger.error(f"Ошибка генерации документа: {e}")
        return False
    return True
