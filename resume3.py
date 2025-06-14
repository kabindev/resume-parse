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
    Validate and clean extracted resume data, including experience and age.

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
        "education": clean_text(extracted_data.get("education", "")),
        "location": clean_text(extracted_data.get("location", "")),
        "summary": clean_text(extracted_data.get("summary", ""))
        # experience_years and date_of_birth will be handled below
    }

    # Experience Processing
    experience_years_str = clean_text(extracted_data.get("experience_years", ""))
    experience_numeric = None
    experience_level = "Fresher"  # Default to Fresher

    if experience_years_str:
        # Try to extract a number (float or int)
        match = re.search(r'(\d+\.?\d*|\.\d+)', experience_years_str)
        if match:
            try:
                val = float(match.group(1))
                experience_numeric = val
                if val >= 1.0:
                    experience_level = "Experienced"
                # If val < 1.0, it remains "Fresher" which is already set
            except ValueError:
                # If conversion fails, treat as unparsable, keep default Fresher or rely on string
                pass # experience_numeric remains None
        elif any(keyword in experience_years_str.lower() for keyword in ["fresher", "none", "0", "n/a"]):
            experience_level = "Fresher"
            experience_numeric = 0.0 # Explicitly set to 0 for freshers if keywords found
        # If string doesn't match numeric regex or fresher keywords, experience_numeric remains None
        # and experience_level remains "Fresher" unless numeric parsing above changes it.

    parsed_info["experience_years_processed"] = experience_numeric if experience_numeric is not None else experience_years_str
    parsed_info["experience_level"] = experience_level

    # Date of Birth and Age Processing
    dob_str = clean_text(extracted_data.get("date_of_birth", ""))
    age_calculated = ""  # Keep as empty string if not calculable
    dob_formatted = dob_str  # Default to original string if parsing fails

    if dob_str:
        common_date_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d/%m/%Y', '%b %d, %Y', '%B %d, %Y',
            '%d %b %Y', '%d %B %Y', # Day Month Year e.g., 20 Jun 1990
            '%Y', # Year only, might not be enough for precise age but can parse
        ]
        for fmt in common_date_formats:
            try:
                dob_dt = datetime.strptime(dob_str, fmt)
                dob_formatted = dob_dt.strftime('%Y-%m-%d') # Standardize format

                # If only year was parsed, age calculation might be off or partial
                if fmt == '%Y': # Handle year-only case, age will be approximate
                    today = datetime.today()
                    age_calculated = today.year - dob_dt.year
                else: # For more complete dates
                    today = datetime.today()
                    age_calculated = today.year - dob_dt.year - ((today.month, today.day) < (dob_dt.month, dob_dt.day))
                break  # Stop if a format matches
            except ValueError:
                continue # Try next format

    parsed_info["date_of_birth_processed"] = dob_formatted
    parsed_info["age_calculated"] = age_calculated

    # Remove original fields if they exist to avoid confusion, as _processed fields are now the source of truth
    if "experience_years" in parsed_info: # This check is important as it might have been populated by AI
        del parsed_info["experience_years"]
    # No need to explicitly check and del extracted_data.get("date_of_birth", "") from parsed_info,
    # as it was never added to parsed_info with that key.
    # The new fields are "date_of_birth_processed".

    # Gender Processing
    raw_gender = clean_text(extracted_data.get("gender", ""))
    processed_gender = raw_gender.lower()
    standardized_gender = "Not Specified"  # Default

    if processed_gender in ["male", "m"]:
        standardized_gender = "Male"
    elif processed_gender in ["female", "f", "fem"]:
        standardized_gender = "Female"
    elif not processed_gender or processed_gender in ["not specified", "n/a", "prefer not to say", "none"]:
        standardized_gender = "Not Specified"
    # If it's something else, it defaults to "Not Specified" as initialized

    parsed_info["gender_processed"] = standardized_gender

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
    "location": "Current city and country",
    "date_of_birth": "Date of Birth (YYYY-MM-DD)",
    "gender": "Gender (Male, Female, or Not Specified)",
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
            logger.info(f"OpenRouter Response Headers: {response.headers}")

            # Extract rate limit headers
            rate_limit_headers_to_check = {
                'limit': ['X-RateLimit-Limit', 'x-ratelimit-limit', 'openrouter-ratelimit-limit'],
                'remaining': ['X-RateLimit-Remaining', 'x-ratelimit-remaining', 'openrouter-ratelimit-remaining'],
                'reset': ['X-RateLimit-Reset', 'x-ratelimit-reset', 'openrouter-ratelimit-reset']
            }
            extracted_rate_limit_info = {}
            for key, header_names in rate_limit_headers_to_check.items():
                for header_name in header_names:
                    # Check with .get() for case-insensitivity and to avoid KeyError
                    header_value = response.headers.get(header_name)
                    if header_value:
                        extracted_rate_limit_info[key] = header_value
                        break  # Found one, no need to check other variants for this key

            if extracted_rate_limit_info:
                st.session_state['rate_limit_info'] = extracted_rate_limit_info
                logger.info(f"OpenRouter Rate Limit Info: {st.session_state['rate_limit_info']}")

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

                # Log rate limit info if available
                log_message = f"Rate limited (429). Waiting {wait_time:.1f} seconds before retry {attempt + 1}/{config.max_retries}."
                if 'rate_limit_info' in st.session_state and isinstance(st.session_state['rate_limit_info'], dict):
                    limit_info = st.session_state['rate_limit_info']
                    log_message += (
                        f" Current Rate Limit Info: Limit: {limit_info.get('limit', 'N/A')}, "
                        f"Remaining: {limit_info.get('remaining', 'N/A')}, "
                        f"Reset: {limit_info.get('reset', 'N/A')}."
                    )
                    st.warning(
                        f"Rate limited (429). "
                        f"Limit: {limit_info.get('limit', 'N/A')}, "
                        f"Remaining: {limit_info.get('remaining', 'N/A')}, "
                        f"Reset: {limit_info.get('reset', 'N/A')}. "
                        f"Waiting {wait_time:.1f} seconds before retry {attempt + 1}/{config.max_retries}"
                    )
                else:
                    st.warning(f"Rate limited (429). Waiting {wait_time:.1f} seconds before retry {attempt + 1}/{config.max_retries}")

                logger.warning(log_message)
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
            "location": {"type": "STRING", "description": "Current city and country of the applicant"},
            "date_of_birth": {"type": "STRING", "description": "Candidate's date of birth (YYYY-MM-DD if available)"},
            "gender": {"type": "STRING", "description": "Gender of the candidate (Male, Female, or Not Specified)"},
            "summary": {"type": "STRING", "description": "Professional summary or objective"}
        },
        "required": ["name", "email", "phone", "skills"],
        "propertyOrdering": ["name", "email", "phone", "skills", "experience_years", "education", "location", "date_of_birth", "gender", "summary"]
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

def create_json_download_data(resume_data_list: List[Dict[str, Any]]) -> bytes:
    """
    Converts a list of resume dictionaries to a JSON byte string for download.

    Args:
        resume_data_list: List of resume data dictionaries.

    Returns:
        bytes: Formatted JSON data as a byte string.
    """
    json_string = json.dumps(resume_data_list, indent=4)
    return json_string.encode('utf-8')

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
                    # Check if existing_df is not None before accessing its properties
                    if existing_df is not None:
                        col_info = pd.DataFrame({
                            'Column': existing_df.columns,
                            'Data Type': [str(dtype) for dtype in existing_df.dtypes],
                            'Non-Null Count': [existing_df[col].count() for col in existing_df.columns],
                            'Sample Values': [str(existing_df[col].dropna().iloc[0]) if not existing_df[col].dropna().empty else 'No data'
for col in existing_df.columns]
                        })
                        st.dataframe(col_info, use_container_width=True)
                    else:
                        st.warning("Could not display column information as the Excel file is not valid.")

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

    # Display OpenRouter Rate Limit Info
    if 'rate_limit_info' in st.session_state and isinstance(st.session_state.rate_limit_info, dict):
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 OpenRouter API Status")
        rate_info = st.session_state.rate_limit_info
        remaining = rate_info.get('remaining', 'N/A')
        limit = rate_info.get('limit', 'N/A')
        reset = rate_info.get('reset', 'N/A') # This is usually a timestamp or seconds

        # Attempt to make 'reset' more human-readable if it's a timestamp
        reset_display = reset
        if isinstance(reset, str) and reset.isdigit(): # If it's a string of digits, assume it's a Unix timestamp
            try:
                reset_datetime = datetime.fromtimestamp(int(reset))
                reset_display = reset_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')
            except ValueError: # If timestamp is out of range or invalid
                pass # Keep original reset value
        elif isinstance(reset, (int, float)): # If it's a number, assume Unix timestamp
            try:
                reset_datetime = datetime.fromtimestamp(reset)
                reset_display = reset_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')
            except ValueError:
                pass


        st.sidebar.metric(label="Requests Remaining", value=str(remaining))
        st.sidebar.caption(f"Request Limit: {limit}")
        st.sidebar.caption(f"Resets: {reset_display}")

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
                current_filename = uploaded_file.name
                final_json_data = {
                    "Filename": current_filename,
                    "Name": parsed_data.get("name", ""),
                    "Email": parsed_data.get("email", ""),
                    "Phone": parsed_data.get("phone", ""),
                    "Skills": parsed_data.get("skills", []),
                    "Experience Years": parsed_data.get("experience_years_processed", ""),
                    "Experience Level": parsed_data.get("experience_level", "Fresher"),
                    "Education": parsed_data.get("education", ""),
                    "Location": parsed_data.get("location", ""),
                    "File Location": f"/path/to/uploaded_resumes/{current_filename}",
                    "Date of Birth": parsed_data.get("date_of_birth_processed", ""),
                    "Age": parsed_data.get("age_calculated", ""),
                    "Gender": parsed_data.get("gender_processed", "Not Specified")
                }
                parsed_resumes.append(final_json_data)

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
                st.error(f"**{failed['filename']}**: {failed['error']}") # Assuming failed items have 'filename'

    # Display and download results
    if parsed_resumes:
        # Create DataFrame
        df = pd.DataFrame(parsed_resumes) # parsed_resumes now contains dicts with final JSON keys

        # Convert Skills list to comma-separated string for display table
        if 'Skills' in df.columns:
            df['skills_display'] = df['Skills'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
        else:
            df['skills_display'] = "" # Ensure column exists even if Skills is missing

        # Display results table
        st.subheader("📋 Parsed Resume Data")
        # Define all possible columns based on the new JSON structure
        display_columns = [
            "Filename", "Name", "Email", "Phone", "skills_display",
            "Experience Years", "Experience Level", "Education", "Location",
            "File Location", "Date of Birth", "Age", "Gender"
        ]
        # Filter for columns actually present in the DataFrame to avoid errors
        display_columns_present = [col for col in display_columns if col in df.columns]

        display_df = df[display_columns_present].copy()
        # Headers for display_df are already user-friendly from display_columns_present

        st.dataframe(display_df, use_container_width=True)

        # Download options
        st.subheader("💾 Download Options")

        # Use more columns for download buttons if appending to existing Excel
        if excel_mode == "Append to Existing File" and existing_excel_file:
            col1, col2, col3 = st.columns(3)
        else:
            col1, col2 = st.columns(2) # Default to two columns

        with col1:
            # Prepare DataFrame for Excel (without skills_display)
            excel_df = df.copy()
            if 'skills_display' in excel_df.columns:
                excel_df.drop('skills_display', axis=1, inplace=True)

            excel_buffer = create_excel_download(excel_df)
            st.download_button(
                label="📥 Download Batch as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"resume_batch_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download all successfully parsed resumes from this batch as an Excel file."
            )

        with col2:
            # JSON Download Button for current batch
            json_download_bytes = create_json_download_data(parsed_resumes) # parsed_resumes is already a list of dicts
            st.download_button(
                label="📥 Download Batch as JSON",
                data=json_download_bytes,
                file_name=f"resume_batch_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download all successfully parsed resumes from this batch as a JSON file."
            )

        if excel_mode == "Append to Existing File" and existing_excel_file:
            with col3: # This column only exists if appending
                # Append to existing Excel file
                updated_excel = append_to_existing_excel(parsed_resumes, existing_excel_file)
                if updated_excel:
                    st.download_button(
                        label="📥 Download Updated Excel File",
                        data=updated_excel.getvalue(),
                        file_name=f"updated_resume_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download Excel file with new batch data appended."
                    )

def display_results():
    """Display stored results from session state."""

    if st.session_state.resume_database and st.session_state.processing_complete:
        st.header("💾 Session Results")

        # Show session statistics
        total_resumes = len(st.session_state.resume_database)
        last_batch = st.session_state.last_processed_count

        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric("📚 Total Resumes in Session", total_resumes)
        with stat_col2:
            st.metric("🆕 Last Batch Processed", last_batch)

        # Display all session data
        if st.button("📊 View All Session Data"):
            df = pd.DataFrame(st.session_state.resume_database) # resume_database stores dicts with final JSON keys

            # Convert Skills list to comma-separated string for display table
            if 'Skills' in df.columns:
                df['skills_display'] = df['Skills'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                )
            else:
                df['skills_display'] = "" # Ensure column exists

            # Define all possible columns based on the new JSON structure
            display_columns = [
                "Filename", "Name", "Email", "Phone", "skills_display",
                "Experience Years", "Experience Level", "Education", "Location",
                "File Location", "Date of Birth", "Age", "Gender"
            ]
            # Filter for columns actually present in the DataFrame
            display_columns_present = [col for col in display_columns if col in df.columns]

            display_df = df[display_columns_present].copy()
            # Headers are already user-friendly

            st.dataframe(display_df, use_container_width=True)

            st.subheader("💾 Download All Session Data")
            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
                # Prepare DataFrame for Excel (without skills_display)
                excel_df_all = df.copy()
                if 'skills_display' in excel_df_all.columns:
                    excel_df_all.drop('skills_display', axis=1, inplace=True)

                excel_buffer_all = create_excel_download(excel_df_all)
                st.download_button(
                    label="📥 Download All as Excel",
                    data=excel_buffer_all.getvalue(),
                    file_name=f"all_session_resume_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download all resumes processed in this session as an Excel file."
                )

            with dl_col2:
                # JSON Download Button for all session data
                json_all_download_bytes = create_json_download_data(st.session_state.resume_database)
                st.download_button(
                    label="📥 Download All as JSON",
                    data=json_all_download_bytes,
                    file_name=f"all_session_resume_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Download all resumes processed in this session as a JSON file."
                )

# ------ APPLICATION ENTRY POINT ------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page and try again.")