import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd
import re
import time

load_dotenv()
GOOGLE_API_KEY = os.getenv("EKAPOL_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# Select the Gemini 2.0 Flash model
model = genai.GenerativeModel('gemini-2.0-flash')

def analyze_translation_quality(source_text, translated_text, max_retries=5, wait_time=60):
    prompt = f"""
    You are an expert in evaluating the quality of machine translation based on the MQM (Multidimensional Quality Metrics) framework.
    Given the following source text and its translation, identify and categorize any errors as critical, major, or minor.

    Error Category Definitions:
    minor - ความหมายถูก เลือกคำไม่ดี (Correct meaning, poor word choice)
    ตัวอย่าง: แปล "บ้านฉัน" เป็น "บ้านผม" (Example: Translates "my house" as "my (male speaker)'s house")

    major - ความหมายผิดเล็กน้อย (Slightly incorrect meaning)
    ตัวอย่าง: แปล "ม้าหลายตัว" เป็น "ม้า" (Example: Translates "several horses" as "horse")

    critical - ความหมายผิดร้ายแรง (Severely incorrect meaning)
    ตัวอย่าง: แปล "ม้า" เป็น "แมว" (Example: Translates "horse" as "cat")

    Source Text: "{source_text}"
    Translated Text: "{translated_text}"

    Identify any errors in the translation based on the source text and categorize them as critical, major, or minor according to the definitions above. Provide the count of each error type.

    Provide the output *ONLY* in the following strict format. Do not include any other text or explanation.

    Example output: Critical: 1, Major: 0, Minor: 2
    """
    retries = 0
    while retries < max_retries:
        try:
            response = model.generate_content(prompt)
            response.resolve()
            return response.text
        except Exception as e:
            error_msg = str(e).lower()
            print(f"Error during API call: {e}")
            if "rate limit" in error_msg or "quota" in error_msg:
                print(f"Rate limit hit. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                retries += 1
            else:
                return f"Error during API call: {e}"
    return "Error: Max retries exceeded due to rate limits."


# def extract_mqm_scores(api_response):
#     scores = {"critical": 0, "major": 0, "minor": 0}
#     lines = api_response.split('\n')
#     for line in lines:
#         if "Critical:" in line:
#             match = re.search(r'Critical:\s*(\d+)', line)
#             if match:
#                 scores["critical"] = int(match.group(1))
#         elif "Major:" in line:
#             match = re.search(r'Major:\s*(\d+)', line)
#             if match:
#                 scores["major"] = int(match.group(1))
#         elif "Minor:" in line:
#             match = re.search(r'Minor:\s*(\d+)', line)
#             if match:
#                 scores["minor"] = int(match.group(1))
#     return scores

def extract_mqm_scores(api_response):
    scores = {"critical": 0, "major": 0, "minor": 0}
    if "Error" in api_response:
        return scores  # Return default scores if there was an error

    match = re.search(r'Critical:\s*(\d+),\s*Major:\s*(\d+),\s*Minor:\s*(\d+)', api_response, re.IGNORECASE)
    if match:
        scores["critical"] = int(match.group(1))
        scores["major"] = int(match.group(2))
        scores["minor"] = int(match.group(3))
    return scores

def process_csv_pandas(csv_filepath):
    df = pd.read_csv(csv_filepath)
    df['gemini_response'] = None
    df['g.critical'] = None  # Initialize new columns to null
    df['g.major'] = None
    df['g.minor'] = None
    for index, row in df.iterrows():
        source_text = row['sourceText']
        translated_text = row['translatedText']
        api_response = analyze_translation_quality(source_text, translated_text)
        print( f"API Response: {api_response}")  # Debugging line 
        mqm_scores = extract_mqm_scores(api_response)

        df.loc[index, 'gemini_response'] = api_response
        df.loc[index, 'g.critical'] = mqm_scores['critical'] 
        df.loc[index, 'g.major'] = mqm_scores['major']
        df.loc[index, 'g.minor'] = mqm_scores['minor']
        print(f"Processed Index: {index}, MQM Scores: {mqm_scores}")
    return df  # Return the modified DataFrame

def write_results_to_csv_pandas(df, output_filepath='cleaned_gemini_claude_mqm.csv'):
    df.to_csv(output_filepath, index=False, encoding='utf-8')
    print(f"Results written to {output_filepath}")

if __name__ == "__main__":
    csv_file = 'cleaned_claude_mqm.csv'
    processed_df = process_csv_pandas(csv_file)
    write_results_to_csv_pandas(processed_df)