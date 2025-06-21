import spacy
import re
import json
from dateparser.search import search_dates
from datetime import datetime
import geonamescache
import sys
import os
import pdfplumber

# Charger le modèle spaCy anglais
nlp = spacy.load('en_core_web_lg')

def load_skill_db(skill_db_path):
    with open(skill_db_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # On suppose que c'est une liste de compétences ou un dict avec une clé principale
    if isinstance(data, dict) and 'skills' in data:
        return set([s.lower() for s in data['skills']])
    elif isinstance(data, list):
        return set([s.lower() for s in data])
    else:
        # fallback: toutes les valeurs string du json
        skills = set()
        def extract_strings(obj):
            if isinstance(obj, str):
                skills.add(obj.lower())
            elif isinstance(obj, list):
                for v in obj:
                    extract_strings(v)
            elif isinstance(obj, dict):
                for v in obj.values():
                    extract_strings(v)
        extract_strings(data)
        return skills

def extract_skills_from_text(text, skill_db):
    found_skills = set()
    text_lower = text.lower()
    for skill in skill_db:
        # On ne matche que des mots/expressions entières
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill)
    return list(found_skills)

def calculate_total_years_experience(text):
    lines = text.splitlines()
    printed_lines = set()
    total_years = 0
    for line in lines:
        line = line.lower()
        line = re.sub(r"\b(months?|years?|mos|yr|yrs|mois|an|ans)\b", "", line, flags=re.IGNORECASE)
        line = line.replace(".", "") 
        line = line.replace("/", " ") 
        line = line.replace("-", " ") 
        line = line.replace("present", datetime.now().strftime("%b %d, %Y"))
        line = line.replace("today", datetime.now().strftime("%b %d, %Y"))
        line = line.replace("now", datetime.now().strftime("%b %d, %Y"))
        line = line.replace("aujourd'hui", datetime.now().strftime("%b %d, %Y"))
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
    return round(total_years, 2)

def detect_address(text):
    gc = geonamescache.GeonamesCache()
    countries = [country['name'] for country in gc.get_countries().values()]
    detected_countries = []
    for country in countries:
        if re.search(r'\b' + re.escape(country.lower()) + r'\b', text.lower()):
            detected_countries.append(country)
    return detected_countries

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_cv_data.py <folder_path> <skill_db_json>")
        sys.exit(1)
    folder_path = sys.argv[1]
    skill_db_path = sys.argv[2]
    skill_db = load_skill_db(skill_db_path)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf') or filename.lower().endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            print(f"\n--- Résultats pour : {filename} ---")
            if filename.lower().endswith('.pdf'):
                cv_text = extract_text_from_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    cv_text = f.read()
            skills = extract_skills_from_text(cv_text, skill_db)
            years_exp = calculate_total_years_experience(cv_text)
            addresses = detect_address(cv_text)
            print("Compétences extraites:", skills)
            print("Années d'expérience totales:", years_exp)
            print("Pays détectés:", addresses)

if __name__ == "__main__":
    main() 