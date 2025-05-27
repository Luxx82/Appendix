import pandas as pd
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

INPUT_FILE = "Prova2023vFinalGPT.xlsx"
OUTPUT_FILE = "gpt41_single_run_output.xlsx"
MODEL = "gpt-4.1"
TEMP = 0.0

# Load spreadsheet and extract questions
df = pd.read_excel(INPUT_FILE)
column_names = df.columns.str.lower().str.strip()
df.columns = column_names

questions = df[[
    "número da questão",
    "comando da questão",
    "alternativa a",
    "alternativa b",
    "alternativa c",
    "alternativa d",
    "alternativa e",
    "contém imagem?",
    "image url"
]]

answers = []

for _, row in questions.iterrows():
    q_num = row["número da questão"]
    question = str(row["comando da questão"])
    tipo = str(row["contém imagem?"]).strip().lower()
    img_url = row.get("image url", None)

    alts = row[["alternativa a", "alternativa b", "alternativa c", "alternativa d", "alternativa e"]].fillna("Alternativa em branco")
    alt_text = (
        f"A: {alts['alternativa a']}\n"
        f"B: {alts['alternativa b']}\n"
        f"C: {alts['alternativa c']}\n"
        f"D: {alts['alternativa d']}\n"
        f"E: {alts['alternativa e']}"
    )

    if tipo == "imagem":
        prefix = "A seguinte pergunta faz referência a uma imagem. Analise o conteúdo da imagem apresentada e, com base nela, responda à pergunta a seguir."
    elif tipo == "video":
        prefix = "A imagem abaixo contém uma sequência de quadros extraídos de um vídeo clínico. Analise a progressão dos eventos representados e responda."
    else:
        prefix = "A seguinte pergunta faz parte de uma prova de cardiologia."

    instruction = (
        "Escolha a alternativa mais provável entre as fornecidas. "
        "Sua resposta deve ser apenas uma letra (A, B, C, D ou E), sem explicações, palavras adicionais ou pontuação. "
        "Mesmo que você não tenha certeza, você deve obrigatoriamente escolher uma das alternativas fornecidas. "
        "A saída deve conter apenas uma única letra maiúscula entre A e E."
    )

    combined_prompt = f"{prefix}\n\n{question}\n\n{alt_text}\n\n{instruction}"

    if isinstance(img_url, str) and img_url.startswith("http"):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url,
                            "detail": "auto"
                        }
                    },
                    {
                        "type": "text",
                        "text": combined_prompt
                    }
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": combined_prompt
            }
        ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMP,
            max_tokens=1
        )
        answer = response.choices[0].message.content.strip().upper()
        if answer not in ["A", "B", "C", "D", "E"]:
            raise ValueError(f"Unexpected answer format: {answer}")
    except Exception as e:
        print(f"❌ Q{q_num}: Failed — {e}")
        answer = "Erro"

    answers.append(answer)
    print(f"✅ Q{q_num}: {answer}")
    time.sleep(0.6)

# Save output
df_output = pd.DataFrame({"Q#": questions["número da questão"], "GPT-4.5 Response": answers})
df_output.to_excel(OUTPUT_FILE, index=False)
print(f"\n✅ All done. Results saved to {OUTPUT_FILE}")
