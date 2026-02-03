import os
from typing import Optional
from app.services.ingestion.interfaces import DocumentParser
from pypdf import PdfReader
from docx import Document
import logging

# Suppress annoying pypdf warnings about rotated text
logging.getLogger("pypdf").setLevel(logging.ERROR)

class PDFParser(DocumentParser):
    def parse(self, file_path: str) -> str:
        text = ""
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    # Improve Layout Awareness
                    # "layout" mode tries to preserve physical layout (good for tables/columns)
                    page_text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False)
                    if page_text:
                        text += f"\n## Page {page_num}\n" + page_text + "\n"
                except KeyError as e:
                    # Handle bbox and other key errors with fallback extraction
                    if 'bbox' in str(e):
                        print(f"  ⚠️ Page {page_num}/{total_pages}: bbox error, trying fallback extraction")
                    else:
                        print(f"  ⚠️ Page {page_num}/{total_pages}: {e}, trying fallback extraction")
                    
                    try:
                        # Fallback: try extracting text with simpler method
                        page_text = page.extract_text(extraction_mode="plain")
                        if page_text:
                            text += page_text + "\n"
                    except Exception as fallback_error:
                        print(f"  ❌ Page {page_num}/{total_pages}: Failed to extract text - {fallback_error}")
                        # Continue to next page instead of failing entire document
                        continue
                except Exception as e:
                    print(f"  ❌ Page {page_num}/{total_pages}: Unexpected error - {e}")
                    continue
            
            if not text.strip():
                raise ValueError(f"No text could be extracted from PDF: {file_path}")
                
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            raise e
        return text

class DocxParser(DocumentParser):
    def parse(self, file_path: str) -> str:
        text = ""
        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            print(f"Error parsing DOCX {file_path}: {e}")
            raise e
        return text

class DocumentParserFactory:
    @staticmethod
    def get_parser(file_path: str) -> DocumentParser:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == ".pdf":
            return PDFParser()
        elif ext == ".docx":
            return DocxParser()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
