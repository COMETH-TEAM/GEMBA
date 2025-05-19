import pandas as pd
import time
import re
import openai

# Use the new client-based interface
client = openai.OpenAI(api_key='put your api key')  # Or set OPENAI_API_KEY as env variable

# Load the CSV
df = pd.read_csv("cleaned_gemini_claude_mqm.csv")

# Prepare new columns
df["o.critical"] = None
df["o.major"] = None
df["o.minor"] = None

# Prompt template
def build_prompt(source_text, translated_text):
    return f"""You are an expert in evaluating the quality of machine translation based on the MQM (Multidimensional Quality Metrics) framework.
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

Format your response as:
Critical: [count]
Major: [count]
Minor: [count]
"""

# Loop through rows
for idx, row in df.iterrows():
    prompt = build_prompt(row["sourceText"], row["translatedText"])

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional MQM evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )

        content = response.choices[0].message.content
        print(f"Response for row {idx}: {content}")

        # Extract counts
        critical = re.search(r"Critical:\s*(\d+)", content)
        major = re.search(r"Major:\s*(\d+)", content)
        minor = re.search(r"Minor:\s*(\d+)", content)

        df.at[idx, "o.critical"] = int(critical.group(1)) if critical else 0
        df.at[idx, "o.major"] = int(major.group(1)) if major else 0
        df.at[idx, "o.minor"] = int(minor.group(1)) if minor else 0

    except Exception as e:
        print(f"Error at row {idx}: {e}")
        df.at[idx, "o.critical"] = df.at[idx, "o.major"] = df.at[idx, "o.minor"] = -1

    # time.sleep(1)

# Save the output
df.to_csv("cleaned_gpt_gemini_claude_mqm.csv", index=False)