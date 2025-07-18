AI Resume Parser

An interactive, intelligent resume parsing tool built with Streamlit, designed for recruiters and HR teams. It extracts structured information from bulk PDF resumes using Gemini/OpenRouter AI APIs and saves the results to an Excel file for easy filtering and decision-making.

FEATURES

- Bulk PDF resume upload with progress tracking
- AI-powered information extraction using Gemini or OpenRouter APIs
- Smart parsing of:
  • Name
  • Email
  • Phone number
  • Skills
  • Years of experience
  • Education
  • Location & Summary
- Data validation and cleaning (e.g., phone & email formatting)
- Duplicate detection (by filename or name + email)
- Export results to Excel (download new or append to existing)
- Session-based tracking of processed files
- API key-based authentication (Gemini or OpenRouter)

HOW TO RUN

1. Clone the repository:
   git clone https://github.com/yourusername/resume_parser_electronJS.git
   cd resume_parser_electronJS

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Streamlit app:
   streamlit run app.py

4. Enter your API Key:
   - Gemini (Google): AIza...
   - OpenRouter: sk-or-v1-...

DEPENDENCIES

- streamlit
- PyMuPDF
- requests
- openpyxl, xlsxwriter
- pandas, re, hashlib

NOTES

- Only supports PDF files for input
- API key is required for parsing resumes
- Best suited for HR workflows involving bulk resume screening and filtering

FUTURE ENHANCEMENTS

- OCR support for scanned PDF resumes
- Add parsing support for DOCX files
- Admin panel for managing, filtering, and exporting parsed data

LICENSE

MIT License © 2025 R Kabin Dev
