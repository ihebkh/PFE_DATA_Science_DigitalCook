# PFE Data Science - DigitalCook CV Parser & Job Matcher

## 📋 Project Overview
This project is an intelligent system designed to automate the extraction of key information from resumes (CVs) and recommend relevant job offers. It leverages Natural Language Processing (NLP) and Machine Learning (ML) techniques to parse CVs, extract skills, detect professional experiences, and match candidates to job descriptions stored in a MongoDB database.

## ✨ Key Features
*   **PDF Parsing & OCR**: Extracts text from PDF resumes using `pdf2image` and `pytesseract` (Tesseract-OCR), supporting both English and French.
*   **Skill Extraction**: Utilizes `SkillNer` and `SpaCy` to identify technical and soft skills from the text.
*   **Experience Detection**: Uses a trained **Random Forest Classifier** to distinguish professional experience sections from other text in the CV.
*   **Information Extraction**: 
    *   **Total Experience**: Calculates total years of experience by parsing dates.
    *   **Location Detection**: Identifies countries and addresses using `geonamescache`.
*   **Job Matching**: 
    *   Translates extracted experiences to French for standardization.
    *   Computes a matching score between the CV and active job offers using **TF-IDF** and **Cosine Similarity**.
    *   Ranks job offers based on text similarity, skill overlap, and experience requirements.

## 🛠️ Technologies Used
*   **Language**: Python
*   **NLP**: `spaCy`, `SkillNer`, `TfidfVectorizer`
*   **Machine Learning**: `scikit-learn` (Random Forest, Logistic Regression, etc.)
*   **Data Handling**: `pandas`, `numpy`
*   **Database**: `MongoDB` (pymongo)
*   **OCR**: `pytesseract`, `pdf2image`
*   **Translation**: `deep_translator`

## ⚙️ Installation

### 1. Prerequisites
*   Python 3.8+
*   [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) installed and added to your system path.
*   [Poppler](http://blog.alivate.com.au/poppler-windows/) (for `pdf2image`).

### 2. Install Python Dependencies
Create a `requirements.txt` file (if not present) and install the following packages:

```bash
pip install pandas spacy skillNer scikit-learn matplotlib seaborn pygeonamescache langdetect deep-translator pymongo pdf2image pytesseract joblib
```

### 3. Download NLP Models
Download the required SpaCy model:
```bash
python -m spacy download en_core_web_lg
```

## 🚀 Usage

### Training the Model (Optional)
If you need to retrain the experience classification model:
1.  Ensure `dataset_final.xlsx` is present.
2.  Run the training function (available in the notebooks or script).

### Running the Analysis
The main script is `output_final.py`. It performs the following steps:
1.  Loads the necessary models (`preprocessor.pkl`, `random_forest_model.pkl`) and skill database (`skill_db_relax_20.json`).
2.  Extracts text from a specified PDF CV.
3.  Detects skills, location, and experience duration.
4.  Classifies experience sentences.
5.  Connects to MongoDB to fetch active job offers.
6.  Matches the CV against job offers and prints the best matches.

**To run the script:**
Update the `RESUME_PATH` and `SKILL_DB_PATH` variables in `output_final.py` if necessary, then execute:

```bash
python output_final.py
```

## 📂 Project Structure
*   `output_final.py`: Main script for parsing CVs and matching jobs.
*   `cv_nlp.ipynb`: Notebook for NLP preprocessing and skill extraction experiments.
*   `cv_ml.ipynb`: Notebook for training the Machine Learning models.
*   `skill_db_relax_20.json`: Database of skills for SkillNer.
*   `dataset_final.xlsx` / `dataset_experiences.xlsx`: Datasets used for training.
*   `random_forest_model.pkl`: Saved Random Forest model for experience classification.
*   `preprocessor.pkl`: Saved data preprocessor pipeline.

## 📝 Notes
*   **Tesseract Path**: Ensure the path to `tesseract.exe` inside `output_final.py` matches your installation path:
    ```python
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ```
*   **MongoDB**: The script connects to a cloud MongoDB instance. Ensure you have internet access and the credentials are valid.
