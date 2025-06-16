import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import os
import json
import requests
import pandas as pd
import re
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import time
from datetime import datetime
import random
import io
from io import BytesIO
import openpyxl
from openpyxl import load_workbook
import hashlib

# Set page config
st.set_page_config(
    page_title="AI Resume Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}
.success-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    margin: 1rem 0;
}
.error-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    margin: 1rem 0;
}
.info-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
    margin: 1rem 0;
}
.stProgress > div > div > div > div {
    background-color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# ---- Configuration ----
class Config:
    """Configuration class for the resume parser."""
    
    def __init__(self):
        # Load API key from environment variable or session state
        self.gemini_api_key = st.session_state.get("api_key", os.getenv("GEMINI_API_KEY", ""))
        
        # For OpenRouter API (if using sk-or-v1- key)
        self.gemini_api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Improved rate limiting configuration
        self.base_request_delay = 8.0  # Increased base delay
        self.max_requests_per_minute = 10  # Reduced further
        self.max_retries = 5  # Increased max retries
        self.exponential_backoff_base = 15  # Longer backoff periods
        self.jitter_range = (1, 3)  # Random jitter to avoid synchronized requests
        
        # File processing
        self.supported_extensions = ['.pdf']
        self.max_file_size_mb = 50
        
        # Output configuration
        self.output_columns = ["filename", "name", "email", "phone", "skills", "experience_years", "education"]

# ---- Setup Logging ----
@st.cache_resource
def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create string stream for logging
    log_stream = io.StringIO()
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    
    return logger

logger = setup_logging()

# ---- NEW FUNCTION: File Tracking Utilities ----
def get_file_hash(file_content: bytes) -> str:
    """Generate a unique hash for file content to track processed files."""
    return hashlib.md5(file_content).hexdigest()

def is_file_already_processed(file_name: str, file_hash: str) -> bool:
    """Check if a file has already been processed."""
    processed_files = st.session_state.get("processed_file_hashes", {})
    
    # Check by filename and hash
    if file_name in processed_files:
        return processed_files[file_name] == file_hash
    
    # Also check if any file with this hash was processed (renamed files)
    return file_hash in processed_files.values()

def mark_file_as_processed(file_name: str, file_hash: str):
    """Mark a file as processed."""
    if "processed_file_hashes" not in st.session_state:
        st.session_state.processed_file_hashes = {}
    
    st.session_state.processed_file_hashes[file_name] = file_hash

def get_new_files_only(uploaded_files) -> List:
    """Filter uploaded files to return only new/unprocessed files."""
    new_files = []
    already_processed = []
    
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.getvalue()
        file_hash = get_file_hash(file_content)
        
        if not is_file_already_processed(uploaded_file.name, file_hash):
            new_files.append(uploaded_file)
        else:
            already_processed.append(uploaded_file.name)
    
    return new_files, already_processed

# ---- PDF Text Extraction Function ----
def extract_text_from_pdf(pdf_file) -> Optional[str]:
    """
    Extracts text from each page of a PDF using PyMuPDF.
    
    Args:
        pdf_file: Uploaded PDF file from Streamlit
    
    Returns:
        Optional[str]: Full text extracted from the PDF, or None if an error occurs.
    """
    try:
        # Check file size
        file_size_mb = len(pdf_file.getvalue()) / (1024 * 1024)
        if file_size_mb > Config().max_file_size_mb:
            st.warning(f"File {pdf_file.name} is too large ({file_size_mb:.1f}MB). Skipping.")
            return None
        
        # Read PDF from bytes
        doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():  # Only add non-empty pages
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        
        doc.close()
        
        if not full_text.strip():
            st.warning(f"No text extracted from {pdf_file.name}")
            return None
        
        return full_text.strip()
        
    except Exception as e:
        st.error(f"Failed to extract text from {pdf_file.name}: {e}")
        return None

# ---- Utility Functions ----
def clean_text(text: str) -> str:
    """Clean and normalize text fields."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text.strip())

def validate_email(email: str) -> str:
    """Validate email format."""
    if not isinstance(email, str):
        return ""
    email = email.strip().lower()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return email if re.match(email_pattern, email) else ""

def clean_phone(phone: str) -> str:
    """Clean and format phone number."""
    if not isinstance(phone, str):
        return ""
    
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    # Return empty if no digits or too short/long
    if len(digits) < 10 or len(digits) > 15:
        return ""
    
    return digits

def clean_skills_list(skills: List[str]) -> List[str]:
    """Clean and deduplicate skills list."""
    if not isinstance(skills, list):
        return []
    
    cleaned_skills = []
    seen_skills = set()
    
    for skill in skills:
        if isinstance(skill, str):
            skill = clean_text(skill)
            if skill and skill.lower() not in seen_skills:
                cleaned_skills.append(skill)
                seen_skills.add(skill.lower())
    
    return cleaned_skills

def validate_and_clean_data(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean extracted resume data.
    
    Args:
        extracted_data: Raw data from AI API
    
    Returns:
        Dict[str, Any]: Cleaned and validated data
    """
    parsed_info = {
        "name": clean_text(extracted_data.get("name", "")),
        "email": validate_email(extracted_data.get("email", "")),
        "phone": clean_phone(extracted_data.get("phone", "")),
        "skills": clean_skills_list(extracted_data.get("skills", [])),
        "experience_years": clean_text(extracted_data.get("experience_years", "")),
        "education": clean_text(extracted_data.get("education", "")),
        "location": clean_text(extracted_data.get("location", "")),
        "summary": clean_text(extracted_data.get("summary", ""))
    }
    
    return parsed_info

def smart_delay(attempt: int = 0) -> None:
    """
    Implement smart delay with jitter and progressive backoff.
    
    Args:
        attempt: Current attempt number (0-based)
    """
    config = Config()
    
    # Base delay with jitter
    jitter = random.uniform(*config.jitter_range)
    base_delay = config.base_request_delay + jitter
    
    # Add progressive delay for retries
    if attempt > 0:
        progressive_delay = attempt * 5  # 5 seconds per retry attempt
        total_delay = base_delay + progressive_delay
    else:
        total_delay = base_delay
    
    time.sleep(total_delay)

# ---- Enhanced AI API Interaction ----
def parse_resume_with_ai(resume_text: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Sends resume text to the AI API for information extraction.
    Works with both Gemini and OpenRouter APIs.
    
    Args:
        resume_text (str): The full text extracted from a resume.
        api_key (str): Your API key.
    
    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the extracted information,
                                or None if the API call fails.
    """
    if not resume_text or not resume_text.strip():
        st.warning("No resume text provided for AI parsing.")
        return None
    
    if not api_key:
        st.error("API key is missing. Please enter your API key in the sidebar.")
        return None
    
    # Check if using OpenRouter API
    is_openrouter = api_key.startswith("sk-or-v1-")
    
    if is_openrouter:
        return parse_with_openrouter(resume_text, api_key)
    else:
        return parse_with_gemini(resume_text, api_key)

def parse_with_openrouter(resume_text: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Parse resume using OpenRouter API with improved error handling and rate limiting."""
    
    # Shorter, more focused prompt to reduce token usage
    prompt = f"""Extract key information from this resume and return as JSON:

{{
    "name": "Full name",
    "email": "Email address", 
    "phone": "Phone number",
    "skills": ["skill1", "skill2"],
    "experience_years": "Years of experience",
    "education": "Highest degree",
    "location": "Location",
    "summary": "Brief summary"
}}

Resume: {resume_text[:6000]}

JSON only:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app-name.com",
        "X-Title": "Resume Parser"
    }
    
    payload = {
        "model": "google/gemini-2.0-flash-exp:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": False
    }
    
    config = Config()
    
    for attempt in range(config.max_retries):
        try:
            # Smart delay with progressive backoff
            smart_delay(attempt)
            
            response = requests.post(
                config.gemini_api_url,
                headers=headers,
                json=payload,
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("choices") and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    
                    # More robust JSON extraction
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                    if json_match:
                        json_string = json_match.group()
                        try:
                            extracted_data = json.loads(json_string)
                            return validate_and_clean_data(extracted_data)
                        except json.JSONDecodeError as je:
                            st.error(f"JSON decode error: {je}")
                    else:
                        st.error("No valid JSON found in response")
                else:
                    st.error(f"Unexpected API response structure")
                    
            elif response.status_code == 429:
                # Rate limiting - implement exponential backoff
                wait_time = config.exponential_backoff_base * (2 ** attempt) + random.uniform(1, 5)
                st.warning(f"Rate limited (429). Waiting {wait_time:.1f} seconds before retry {attempt + 1}/{config.max_retries}")
                time.sleep(wait_time)
                continue
                
            elif response.status_code == 402:
                st.error("Payment required (402). Check your API credits.")
                return None
                
            elif response.status_code == 401:
                st.error("Unauthorized (401). Check your API key.")
                return None
                
            else:
                st.error(f"API request failed with status {response.status_code}")
                
        except requests.exceptions.Timeout:
            st.error(f"Request timed out (attempt {attempt + 1}/{config.max_retries})")
            if attempt == config.max_retries - 1:
                return None
                
        except requests.exceptions.ConnectionError as e:
            st.error(f"Connection error (attempt {attempt + 1}/{config.max_retries})")
            if attempt == config.max_retries - 1:
                return None
            time.sleep(10)
            
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return None
    
    st.error("All retry attempts exhausted")
    return None

def parse_with_gemini(resume_text: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Sends resume text to the Gemini API for information extraction.
    """
    json_schema = {
        "type": "OBJECT",
        "properties": {
            "name": {"type": "STRING", "description": "Full name of the candidate"},
            "email": {"type": "STRING", "description": "Primary email address"},
            "phone": {"type": "STRING", "description": "Primary phone number"},
            "skills": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "List of technical and professional skills"
            },
            "experience_years": {"type": "STRING", "description": "Total years of professional experience"},
            "education": {"type": "STRING", "description": "Highest degree or relevant education"},
            "location": {"type": "STRING", "description": "Current location or address"},
            "summary": {"type": "STRING", "description": "Professional summary or objective"}
        },
        "required": ["name", "email", "phone", "skills"],
        "propertyOrdering": ["name", "email", "phone", "skills", "experience_years", "education", "location", "summary"]
    }
    
    prompt = f"""
Extract resume information and return as JSON:

Resume Text:
{resume_text[:8000]}

JSON Schema:
{json.dumps(json_schema, indent=2)}

Respond with ONLY the JSON object:
"""

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": json_schema,
            "temperature": 0.1,
            "maxOutputTokens": 2048
        }
    }
    
    try:
        smart_delay()
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        if (result.get("candidates") and 
            len(result["candidates"]) > 0 and 
            result["candidates"][0].get("content") and 
            result["candidates"][0]["content"].get("parts")):
            
            json_string = result["candidates"][0]["content"]["parts"][0]["text"]
            extracted_data = json.loads(json_string)
            return validate_and_clean_data(extracted_data)
        else:
            st.warning("Gemini API response missing expected content structure")
            return None
            
    except Exception as e:
        st.error(f"Error during Gemini parsing: {e}")
        return None

# ---- Data Management Functions ----
def check_duplicate_resume(new_data: Dict[str, Any], existing_df: pd.DataFrame) -> bool:
    """
    Check if a resume already exists in the database.
    
    Args:
        new_data: New resume data to check
        existing_df: Existing DataFrame
    
    Returns:
        bool: True if duplicate found
    """
    if existing_df.empty:
        return False
    
    # Check by filename first
    if 'filename' in existing_df.columns:
        if new_data['filename'] in existing_df['filename'].values:
            return True
    
    # Check by name and email combination
    if 'name' in existing_df.columns and 'email' in existing_df.columns:
        name_match = existing_df['name'] == new_data.get('name', '')
        email_match = existing_df['email'] == new_data.get('email', '')
        if any(name_match & email_match):
            return True
    
    return False

def create_excel_download(df: pd.DataFrame) -> BytesIO:
    """Create Excel file in memory for download."""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Resume_Database', index=False)
        
        # Get workbook and worksheet objects for formatting
        workbook = writer.book
        worksheet = writer.sheets['Resume_Database']
        
        # Add formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': "#9FEA09",
            'border': 1
        })
        
        # Apply header formatting
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Auto-adjust column widths
        for i, col in enumerate(df.columns):
            max_length = max(df[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, min(max_length + 2, 50))
    
    output.seek(0)
    return output

# ---- NEW FUNCTION: Excel Append Functionality ----
def load_existing_excel(uploaded_excel_file) -> Optional[pd.DataFrame]:
    """
    Load existing Excel file for appending data.
    
    Args:
        uploaded_excel_file: Streamlit uploaded file object
    
    Returns:
        Optional[pd.DataFrame]: Existing DataFrame or None if error
    """
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_excel_file, engine='openpyxl')
        
        # Validate that it has the expected columns
        expected_columns = ['filename', 'name', 'email', 'phone', 'skills', 'experience_years', 'education']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            st.warning(f"Excel file is missing required columns: {', '.join(missing_columns)}")
            st.info("Expected columns: " + ", ".join(expected_columns))
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

def append_to_existing_excel(new_data: List[Dict[str, Any]], existing_excel_file) -> Optional[BytesIO]:
    """
    Append new resume data to an existing Excel file.
    
    Args:
        new_data: List of new resume data dictionaries
        existing_excel_file: Uploaded Excel file
    
    Returns:
        Optional[BytesIO]: Updated Excel file in memory or None if error
    """
    try:
        # Load existing data
        existing_df = load_existing_excel(existing_excel_file)
        if existing_df is None:
            return None
        
        # Convert new data to DataFrame
        new_df = pd.DataFrame(new_data)
        
        # Convert skills list to comma-separated string for Excel
        if 'skills' in new_df.columns:
            new_df['skills'] = new_df['skills'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
        
        # Also convert existing skills if they're not already strings
        if 'skills' in existing_df.columns:
            existing_df['skills'] = existing_df['skills'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
        
        # Remove duplicates from new data
        filtered_new_data = []
        duplicates_found = 0
        
        for _, new_row in new_df.iterrows():
            is_duplicate = False
            
            # Check for duplicates by filename
            if new_row['filename'] in existing_df['filename'].values:
                is_duplicate = True
                duplicates_found += 1
            
            # Check for duplicates by name and email combination
            elif not is_duplicate:
                name_matches = existing_df['name'] == new_row['name']
                email_matches = existing_df['email'] == new_row['email']
                if any(name_matches & email_matches) and new_row['name'] and new_row['email']:
                    is_duplicate = True
                    duplicates_found += 1
            
            if not is_duplicate:
                filtered_new_data.append(new_row.to_dict())
        
        if duplicates_found > 0:
            st.warning(f"⚠️ {duplicates_found} duplicate(s) found and skipped")
        
        if not filtered_new_data:
            st.warning("No new data to append - all records were duplicates")
            return None
        
        # Create final DataFrame
        filtered_new_df = pd.DataFrame(filtered_new_data)
        final_df = pd.concat([existing_df, filtered_new_df], ignore_index=True)
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            final_df.to_excel(writer, sheet_name='Resume_Database', index=False)
            
            # Get workbook and worksheet objects for formatting
            workbook = writer.book
            worksheet = writer.sheets['Resume_Database']
            
            # Add formatting
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': "#9FEA09",
                'border': 1
            })
            
            # Highlight new rows
            new_row_format = workbook.add_format({
                'bg_color': '#E8F4FD',
                'border': 1
            })
            
            # Apply header formatting
            for col_num, value in enumerate(final_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Highlight new rows (starting from the row after existing data)
            start_new_row = len(existing_df) + 1  # +1 for header
            for row_num in range(start_new_row, len(final_df) + 1):
                for col_num in range(len(final_df.columns)):
                    cell_value = final_df.iloc[row_num - 1, col_num]
                    worksheet.write(row_num, col_num, cell_value, new_row_format)
            
            # Auto-adjust column widths
            for i, col in enumerate(final_df.columns):
                max_length = max(final_df[col].astype(str).map(len).max(), len(col))
                worksheet.set_column(i, i, min(max_length + 2, 50))
        
        output.seek(0)
        st.success(f"✅ Successfully appended {len(filtered_new_data)} new records to the existing Excel file")
        return output
        
    except Exception as e:
        st.error(f"Error appending to Excel file: {e}")
        return None

def display_excel_append_section():
    """Display the Excel append functionality section."""
    st.header("📊 Append to Existing Excel Database")
    
    st.markdown("""
    <div class="info-box">
    <h4>📋 How to use Excel Append:</h4>
    <ol>
    <li>Upload your existing Excel file (must contain the standard resume columns)</li>
    <li>Process new resume files as usual</li>
    <li>Download the updated Excel file with new data appended</li>
    </ol>
    <p><strong>Note:</strong> Duplicate entries will be automatically detected and skipped.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader for existing Excel
    existing_excel = st.file_uploader(
        "📈 Upload Existing Excel Database (Optional)",
        type=['xlsx', 'xls'],
        help="Upload your existing resume database Excel file. New data will be appended to this file."
    )
    
    if existing_excel:
        # Validate the Excel file
        existing_df = load_existing_excel(existing_excel)
        if existing_df is not None:
            st.success(f"✅ Excel file loaded successfully! Found {len(existing_df)} existing records.")
            
            # Show preview of existing data
            with st.expander("👀 Preview Existing Data", expanded=False):
                st.dataframe(existing_df.head(10), use_container_width=True)
                if len(existing_df) > 10:
                    st.info(f"Showing first 10 records out of {len(existing_df)} total records")
    
    return existing_excel

# ---- NEW FUNCTION: Excel File Location Specification ----
def display_excel_location_selector():
    """Display section for users to specify Excel file location."""
    st.header("📍 Specify Excel File Location")
    
    st.markdown("""
    <div class="info-box">
    <h4>💾 Excel File Location Options:</h4>
    <p>Choose how you want to handle the output Excel file:</p>
    <ul>
    <li><strong>Download Only:</strong> Process resumes and download a new Excel file</li>
    <li><strong>Append to Existing:</strong> Add new data to an existing Excel file</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Radio button for selecting mode
    excel_mode = st.radio(
        "📂 Choose Excel handling mode:",
        options=["Download New File", "Append to Existing File"],
        index=0,
        help="Select whether to create a new Excel file or append to an existing one"
    )
    
    existing_excel_file = None
    if excel_mode == "Append to Existing File":
        st.subheader("📤 Upload Existing Excel File")
        existing_excel_file = st.file_uploader(
            "Choose your existing Excel file",
            type=['xlsx', 'xls'],
            help="Upload the Excel file you want to append new resume data to",
            key="excel_location_uploader"
        )
        
        if existing_excel_file:
            # Validate and preview the existing Excel file
            existing_df = load_existing_excel(existing_excel_file)
            if existing_df is not None:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.success(f"✅ Valid Excel file loaded with {len(existing_df)} existing records")
                with col2:
                    if st.button("👀 Preview Data", key="preview_existing"):
                        st.session_state.show_preview = True
                
                # Show preview if button clicked
                if st.session_state.get('show_preview', False):
                    with st.expander("📊 Existing Data Preview", expanded=True):
                        st.dataframe(existing_df.head(10), use_container_width=True)
                        if len(existing_df) > 10:
                            st.info(f"Showing first 10 out of {len(existing_df)} total records")
                
                    # Show column information
                    st.subheader("📋 Column Information")
                    col_info = pd.DataFrame({
                        'Column': existing_df.columns,
                        'Data Type': [str(dtype) for dtype in existing_df.dtypes],
                        'Non-Null Count': [existing_df[col].count() for col in existing_df.columns],
                        'Sample Values': [str(existing_df[col].dropna().iloc[0]) if not existing_df[col].dropna().empty else 'No data'
for col in existing_df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
    
    return existing_excel_file, excel_mode

# ------ MAIN APPLICATION ------

def main():
    """Main application function with enhanced file tracking."""
    
    # Initialize session state variables
    if 'processed_file_hashes' not in st.session_state:
        st.session_state.processed_file_hashes = {}
    if 'resume_database' not in st.session_state:
        st.session_state.resume_database = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'last_processed_count' not in st.session_state:
        st.session_state.last_processed_count = 0
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 AI Resume Parser</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API key input
    api_key = st.sidebar.text_input(
        "🔑 Enter your API Key:",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="Enter your Gemini API key or OpenRouter API key (starting with sk-or-v1-)"
    )
    
    if api_key:
        st.session_state.api_key = api_key
        if api_key.startswith("sk-or-v1-"):
            st.sidebar.success("✅ OpenRouter API key detected")
        else:
            st.sidebar.success("✅ Gemini API key detected")
    else:
        st.sidebar.warning("⚠️ Please enter your API key to continue")
    
    # Display processing statistics
    if st.session_state.processed_file_hashes:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Processing Statistics")
        st.sidebar.info(f"Files processed this session: {len(st.session_state.processed_file_hashes)}")
        
        # Show recently processed files
        if st.sidebar.button("🗂️ Show Processed Files"):
            with st.sidebar.expander("Recently Processed Files", expanded=True):
                for filename in list(st.session_state.processed_file_hashes.keys())[-5:]:
                    st.write(f"• {filename}")
    
    # Clear session button
    if st.sidebar.button("🗑️ Clear Session Data", help="Clear all processed file tracking"):
        st.session_state.processed_file_hashes = {}
        st.session_state.resume_database = []
        st.session_state.processing_complete = False
        st.session_state.last_processed_count = 0
        st.sidebar.success("✅ Session data cleared!")
        st.rerun()
    
    # Excel file location section
    existing_excel_file, excel_mode = display_excel_location_selector()
    
    # File upload section
    st.header("📂 Upload Resume Files")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files to parse",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF resume files. Only new files will be processed."
    )
    
    if uploaded_files:
        # Filter to get only new files
        new_files, already_processed = get_new_files_only(uploaded_files)
        
        # Display file status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Total Uploaded", len(uploaded_files))
        with col2:
            st.metric("🆕 New Files", len(new_files))
        with col3:
            st.metric("✅ Already Processed", len(already_processed))
        
        # Show already processed files
        if already_processed:
            with st.expander("ℹ️ Already Processed Files (Skipped)", expanded=False):
                for filename in already_processed:
                    st.write(f"• {filename}")
                st.info("These files were already processed in this session and will be skipped.")
        
        # Show new files to be processed
        if new_files:
            with st.expander("🆕 New Files to Process", expanded=True):
                for uploaded_file in new_files:
                    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                    st.write(f"• {uploaded_file.name} ({file_size:.1f} MB)")
        
        # Processing section
        if new_files and api_key:
            if st.button(f"🚀 Process {len(new_files)} New Resume(s)", type="primary"):
                process_resumes(new_files, api_key, existing_excel_file, excel_mode)
        elif not new_files:
            st.info("🔄 All uploaded files have already been processed. Upload new files to continue.")
        elif not api_key:
            st.warning("⚠️ Please enter your API key in the sidebar to process resumes.")
    
    # Display results section
    display_results()

def process_resumes(new_files, api_key, existing_excel_file, excel_mode):
    """Process only new resume files."""
    
    config = Config()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()
    
    parsed_resumes = []
    failed_files = []
    
    total_files = len(new_files)
    
    for i, uploaded_file in enumerate(new_files):
        try:
            # Update progress
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name} ({i + 1}/{total_files})")
            
            # Extract text from PDF
            resume_text = extract_text_from_pdf(uploaded_file)
            
            if not resume_text:
                failed_files.append({
                    'filename': uploaded_file.name,
                    'error': 'Failed to extract text'
                })
                continue
            
            # Parse resume with AI
            parsed_data = parse_resume_with_ai(resume_text, api_key)
            
            if parsed_data:
                # Add filename to parsed data
                parsed_data['filename'] = uploaded_file.name
                parsed_resumes.append(parsed_data)
                
                # Mark file as processed
                file_content = uploaded_file.getvalue()
                file_hash = get_file_hash(file_content)
                mark_file_as_processed(uploaded_file.name, file_hash)
                
                # Show success for current file
                with results_container.container():
                    st.success(f"✅ Successfully processed: {uploaded_file.name}")
            else:
                failed_files.append({
                    'filename': uploaded_file.name,
                    'error': 'AI parsing failed'
                })
        
        except Exception as e:
            failed_files.append({
                'filename': uploaded_file.name,
                'error': str(e)
            })
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
    
    # Complete processing
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    # Store results in session state
    if parsed_resumes:
        st.session_state.resume_database.extend(parsed_resumes)
        st.session_state.processing_complete = True
        st.session_state.last_processed_count = len(parsed_resumes)
    
    # Display final results
    display_processing_results(parsed_resumes, failed_files, existing_excel_file, excel_mode)

def display_processing_results(parsed_resumes, failed_files, existing_excel_file, excel_mode):
    """Display the results of resume processing."""
    
    st.header("📊 Processing Results")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Successfully Parsed", len(parsed_resumes))
    with col2:
        st.metric("❌ Failed", len(failed_files))
    with col3:
        success_rate = (len(parsed_resumes) / (len(parsed_resumes) + len(failed_files)) * 100) if (len(parsed_resumes) + len(failed_files)) > 0 else 0
        st.metric("📈 Success Rate", f"{success_rate:.1f}%")
    
    # Show failed files if any
    if failed_files:
        with st.expander("❌ Failed Files", expanded=False):
            for failed in failed_files:
                st.error(f"**{failed['filename']}**: {failed['error']}")
    
    # Display and download results
    if parsed_resumes:
        # Create DataFrame
        df = pd.DataFrame(parsed_resumes)
        
        # Convert skills list to comma-separated string for display
        if 'skills' in df.columns:
            df['skills_display'] = df['skills'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
        
        # Display results table
        st.subheader("📋 Parsed Resume Data")
        display_columns = ['filename', 'name', 'email', 'phone', 'skills_display', 'experience_years', 'education']
        display_df = df[display_columns].copy()
        display_df.columns = ['Filename', 'Name', 'Email', 'Phone', 'Skills', 'Experience Years', 'Education']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download options
        st.subheader("💾 Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download new Excel file
            excel_buffer = create_excel_download(df.drop('skills_display', axis=1, errors='ignore'))
            st.download_button(
                label="📥 Download as New Excel File",
                data=excel_buffer.getvalue(),
                file_name=f"resume_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download all processed resumes as a new Excel file"
            )
        
        with col2:
            # Append to existing Excel file
            if excel_mode == "Append to Existing File" and existing_excel_file:
                updated_excel = append_to_existing_excel(parsed_resumes, existing_excel_file)
                if updated_excel:
                    st.download_button(
                        label="📥 Download Updated Excel File",
                        data=updated_excel.getvalue(),
                        file_name=f"updated_resume_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download Excel file with new data appended"
                    )

def display_results():
    """Display stored results from session state."""
    
    if st.session_state.resume_database and st.session_state.processing_complete:
        st.header("💾 Session Results")
        
        # Show session statistics
        total_resumes = len(st.session_state.resume_database)
        last_batch = st.session_state.last_processed_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📚 Total Resumes in Session", total_resumes)
        with col2:
            st.metric("🆕 Last Batch Processed", last_batch)
        
        # Display all session data
        if st.button("📊 View All Session Data"):
            df = pd.DataFrame(st.session_state.resume_database)
            
            # Convert skills for display
            if 'skills' in df.columns:
                df['skills_display'] = df['skills'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                )
            
            display_columns = ['filename', 'name', 'email', 'phone', 'skills_display', 'experience_years', 'education']
            display_df = df[display_columns].copy()
            display_df.columns = ['Filename', 'Name', 'Email', 'Phone', 'Skills', 'Experience Years', 'Education']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download all session data
            excel_buffer = create_excel_download(df.drop('skills_display', axis=1, errors='ignore'))
            st.download_button(
                label="📥 Download All Session Data",
                data=excel_buffer.getvalue(),
                file_name=f"complete_resume_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ------ APPLICATION ENTRY POINT ------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page and try again.")
