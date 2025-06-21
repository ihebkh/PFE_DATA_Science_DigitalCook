import spacy
from spacy.matcher import Matcher
import re
from datetime import datetime
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
from spacy import displacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from dateparser.search import search_dates
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from tabulate import tabulate
from collections import defaultdict
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import fitz  # PyMuPDF
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
stop_words = STOP_WORDS

# Load spaCy model
nlp = spacy.load('en_core_web_lg')
matcher = Matcher(nlp.vocab)

# Path to resume file
resume_path = r"C:\Users\khmir\Desktop\cvs\khmiri_iheb_tun_fr.pdf"

# Extracting text from PDF using PyMuPDF
with fitz.open(resume_path) as doc:
    cc = ""
    for page in doc:
        cc += page.get_text()

# Process the text with spaCy
doc = nlp(cc)

# Print the first 1000 characters of the processed text (you can adjust the size)
print(doc[:1000])

# Define various date patterns
date_patterns = [
    [{"TEXT": {"regex": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b"}}],  # MM/DD/YYYY, MM-DD-YYYY, or MM.DD.YYYY
    [{"TEXT": {"regex": r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"}, "OP": "?"}, 
      {"TEXT": ",", "OP": "?"}, 
      {"IS_DIGIT": True, "OP": "?"}, 
      {"TEXT": {"regex": r"\d{4}"}}],  # Month Day, Year format (e.g., Jan 2020)
    [{"TEXT": {"regex": r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}"}}],  # Month Year (e.g., Jan 2020)
    [{"TEXT": {"regex": r"\b\d{4}\b"}}],  # Year only (e.g., 2020)
    [{"TEXT": {"regex": r"\b\d{1,2}[/-]\d{4}\b"}}],  # Month-Year format (e.g., 12/2020 or 12-2020)
    [{"TEXT": {"regex": r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}"}}]  # Month-Day format (e.g., Jan 5)
]

# Add the date patterns to the matcher
for pattern in date_patterns:
    matcher.add("DATE_PATTERN", [pattern], on_match=None)

# Apply the matcher to the document
matches = matcher(doc)

# Extract matched date patterns and print
matched_dates = []
for match_id, start, end in matches:
    span = doc[start:end]  # Get the matched span of text
    matched_dates.append(span.text)

# Display the matched dates
print("Matched Dates:", matched_dates)

# Function to calculate total years of experience
def calculate_total_years_experience(resume_path):
    # Extracting text from PDF again (correcting the previous mistake)
    with fitz.open(resume_path) as doc:
        cc = ""
        for page in doc:
            cc += page.get_text()

    lines = cc.splitlines()
    printed_lines = set()
    total_years = 0

    # Loop through each line and try to calculate the experience
    for line in lines:
        line = line.lower()
        line = re.sub(r"\b(months?|years?|mos|yr|yrs|mois|an|ans)\b", "", line, flags=re.IGNORECASE)
        line = line.replace(".", " ") 
        line = line.replace("present", datetime.now().strftime("%b %d, %Y"))
        line = line.replace("today", datetime.now().strftime("%b %d, %Y"))
        line = line.replace("now", datetime.now().strftime("%b %d, %Y"))
        line = line.replace("aujourd'hui", datetime.now().strftime("%b %d, %Y"))

        # Process each line with spaCy
        doc = nlp(line)
        parsed_date = search_dates(line, languages=["fr", "en"])

        if parsed_date is not None:
            parsed_date = [date[1] for date in parsed_date]

        for ent in doc.ents:
            if ent.label_ == "DATE" and line not in printed_lines:
                if parsed_date is not None and len(parsed_date) >= 2:
                    parsed_date.sort()
                    date1, date2 = parsed_date[:2]
                    diff_years = date2.year - date1.year + float(date2.month - date1.month) / 12
                    total_years += diff_years
                    printed_lines.add(line)
                    print(f"line = {line} ")
    return round(total_years, 2)

# Get total years of experience
total_years_experiences = calculate_total_years_experience(resume_path)
print(f"Total years of experience: {total_years_experiences}")
