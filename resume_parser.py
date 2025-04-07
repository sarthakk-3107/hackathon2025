import fitz  # PyMuPDF
import re
import io

def extract_resume_text(file_obj):
    """
    Extracts raw text from a PDF resume.
    
    Args:
        file_obj (bytes or file-like): PDF content in bytes or a file-like object.

    Returns:
        str: Cleaned plain text from all pages.
    """
    try:
        if isinstance(file_obj, bytes):
            file_obj = io.BytesIO(file_obj)
        file_obj.seek(0)
        content = file_obj.read()
        if not content:
            return "ERROR_PARSING_PDF: Uploaded file is empty."
        doc = fitz.open("pdf", content)
        text = ""
        for page in doc:
            page_text = page.get_text()
            if isinstance(page_text, str):
                text += page_text + " "
            elif isinstance(page_text, list):
                text += " ".join(str(item) for item in page_text)
        return text.strip()
    except Exception as e:
        return f"ERROR_PARSING_PDF: {e}"

def extract_skills(text):
    """
    Dynamically extracts individual skills from the SKILLS or TECHNICAL SKILLS section of a resume.
    Handles subcategories like Programming Languages, Frameworks, Tools, etc.

    Args:
        text (str): Full resume text.

    Returns:
        tuple: (list of cleaned skill strings, raw skills section text)
    """
    # Extract SKILLS section
    match = re.search(
        r"(SKILLS|TECHNICAL SKILLS)(.*?)(WORK EXPERIENCE|EXPERIENCE|EDUCATION|PROJECTS|CERTIFICATIONS|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    section_text = match.group(2).strip() if match else ""

    # Remove subcategory labels
    section_text = re.sub(
        r"(Programming Languages|Software & Packages|Frameworks & Libraries|Data Analysis & Visualization|"
        r"Algorithms & Models|NLP & Prompt Engineering|Tools & Platforms|Hard Skills|Soft Skills|AWS|Certification)\s*:",
        "", section_text, flags=re.IGNORECASE
    )

    # Split using common delimiters
    raw_skills = re.split(r"[,\n•●\-–]", section_text)

    # Clean up and filter
    cleaned_skills = [
        skill.strip()
        for skill in raw_skills
        if len(skill.strip()) > 1 and not skill.strip().isdigit()
    ]

    # Remove duplicates (case-insensitive), preserve order
    seen = set()
    unique_skills = []
    for skill in cleaned_skills:
        key = skill.lower()
        if key not in seen:
            seen.add(key)
            unique_skills.append(skill)

    return unique_skills, section_text

import re
import fitz  # PyMuPDF

def extract_experience(file_bytes):
    """
    Ultra-resilient experience extractor using:
    1. Section-based parsing
    2. Timeline detection
    3. Bullet heuristics
    4. BOLD font segments before first bullet
    """

    def extract_resume_text(file_bytes):
        return "\n".join(
            page.get_text() for page in fitz.open("pdf", file_bytes)
        )

    def extract_primary(text):
        entries = []
        match = re.search(
            r"(WORK EXPERIENCE|EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT HISTORY|CAREER HISTORY)"
            r"(.*?)(EDUCATION|PROJECTS|SKILLS|CERTIFICATIONS|REFERENCES|$)",
            text,
            re.DOTALL | re.IGNORECASE
        )
        if not match:
            return []

        section_text = match.group(2).strip()
        lines = [line.strip() for line in section_text.split("\n") if line.strip()]
        i = 0
        while i < len(lines) - 1:
            line1 = lines[i]
            line2 = lines[i + 1]
            timeline_match = re.search(
                r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s*[-–to]+\s*"
                r"(Present|[A-Za-z]+\s+\d{4})",
                line2, re.IGNORECASE
            )
            if timeline_match:
                position = line1
                timeline = timeline_match.group(0)
                company = re.sub(timeline, "", line2).strip(" ,:-")
                entries.append({
                    "position": position,
                    "company": company,
                    "timeline": timeline
                })
                i += 2
            else:
                i += 1
        return entries

    def extract_by_timeline(text):
        entries = []
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for i, line in enumerate(lines):
            timeline_match = re.search(
                r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s*[-–to]+\s*"
                r"(Present|[A-Za-z]+\s+\d{4})",
                line, re.IGNORECASE
            )
            if timeline_match:
                timeline = timeline_match.group(0)
                position = lines[i - 1] if i - 1 >= 0 else "Unknown"
                company = lines[i - 2] if i - 2 >= 0 else "Unknown"
                entries.append({
                    "position": position,
                    "company": company,
                    "timeline": timeline
                })
        return entries

    def extract_by_bullets(text):
        entries = []
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if re.match(r"^[•●\-]", line):
                position = lines[i - 1].strip() if i - 1 >= 0 else "Unknown"
                company = lines[i - 2].strip() if i - 2 >= 0 else "Unknown"
                entries.append({
                    "position": position,
                    "company": company,
                    "timeline": "N/A"
                })
                while i < len(lines) and re.match(r"^[•●\-]", lines[i].strip()):
                    i += 1
            else:
                i += 1
        return entries

    def extract_bold_before_bullets(file_bytes):
        doc = fitz.open("pdf", file_bytes)
        bold_items = []
        found_bullet = False

        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        font = span.get("font", "").lower()

                        if not text:
                            continue

                        if any(b in text for b in ["•", "●", "-"]):
                            found_bullet = True
                            break

                        if not found_bullet and ("bold" in font or "bd" in font):
                            bold_items.append(text)
                    if found_bullet:
                        break
                if found_bullet:
                    break
            if found_bullet:
                break

        # Map every 2-3 bold items as a role
        entries = []
        for i in range(0, len(bold_items) - 2, 2):
            position = bold_items[i]
            company = bold_items[i + 1]
            timeline = bold_items[i + 2] if i + 2 < len(bold_items) else "N/A"
            entries.append({
                "position": position,
                "company": company,
                "timeline": timeline
            })
        return entries

    # ==== Run multi-stage extraction ====
    text = extract_resume_text(file_bytes)
    experience = extract_primary(text)

    if not experience:
        experience = extract_by_timeline(text)
    if not experience:
        experience = extract_by_bullets(text)
    if not experience:
        experience = extract_bold_before_bullets(file_bytes)

    return experience



def extract_education(text):
    """
    Extracts the education section from resume text.

    Returns:
        str: Education section or "N/A"
    """
    education_section = re.search(r"EDUCATION(.*?)(SKILLS|WORK EXPERIENCE|$)", text, re.DOTALL | re.IGNORECASE)
    return education_section.group(1).strip() if education_section else "N/A"

def extract_projects(text):
    """
    Extracts the projects section from resume text.

    Returns:
        str: Projects section or "N/A"
    """
    projects = re.search(r"ACADEMIC PROJECTS(.*?)(REVIEW PAPER|EXTRACURRICULAR|$)", text, re.DOTALL | re.IGNORECASE)
    return projects.group(1).strip() if projects else "N/A"

# Optional CLI for testing
if __name__ == "__main__":
    pdf_path = input("Enter the path to your resume PDF: ").strip()
    try:
        with open(pdf_path, "rb") as f:
            file_bytes = f.read()

        text = extract_resume_text(file_bytes)
        print("\n📄 Resume Summary")
        print("🎓 Education:\n", extract_education(text), "\n")
        print("💼 Experience:\n", extract_experience(text), "\n")
        skills, _ = extract_skills(text)
        print("🛠️ Skills:\n", ", ".join(skills), "\n")
        print("📂 Projects:\n", extract_projects(text))

    except Exception as e:
        print("❌ Error processing the resume:", e)
