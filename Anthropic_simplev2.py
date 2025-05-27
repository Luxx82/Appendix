import pandas as pd
import time
import os
from anthropic import Anthropic
from dotenv import load_dotenv
import json

load_dotenv()

client = Anthropic()

INPUT_FILE = "Prova2023vFinalGPT.xlsx"
OUTPUT_FILE = "anthropic_image_enabled_output.xlsx"
MODEL = "claude-3-7-sonnet-20250219"
NUM_RUNS = 40

# Load questions
df = pd.read_excel(INPUT_FILE)
column_names = df.columns.str.lower().str.strip()

# Normalize columns for consistency
col_map = {col: name for col, name in zip(df.columns, column_names)}
df.rename(columns=col_map, inplace=True)

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

matrix = pd.DataFrame({"Q#": questions["número da questão"]})

# Add metadata rows
top_rows = pd.DataFrame([
    ["model"] + [MODEL]*NUM_RUNS,
    ["temperature"] + [t for t in ([0.0]*10 + [0.3]*10 + [0.6]*10 + [0.9]*10)],
    ["modality"] + ["simple"]*NUM_RUNS
])
matrix = pd.concat([top_rows.T, matrix], ignore_index=True)

for run in range(1, NUM_RUNS + 1):
    TEMP = [0.0]*10 + [0.3]*10 + [0.6]*10 + [0.9]*10
    temp = TEMP[run - 1]
    print(f"\n🤖 Starting Run {run} — Temp: {temp}")
    run_answers = []

    for _, row in questions.iterrows():
        q_num = row["número da questão"]
        question = str(row["comando da questão"])
        tipo = str(row["contém imagem?"], ).strip().lower()
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
            "A saída deve conter apenas uma única letra maiúscula entre A e E. A saída deve SEMPRE começar pela alternativa escolhida."
        )

        claude_content = []

        # Add image as URL if applicable
        if isinstance(img_url, str) and img_url.startswith("http"):
            claude_content.append({
                "type": "image",
                "source": {
                    "type": "url",
                    "url": img_url
                }
            })

        full_prompt = f"{prefix}\n\n{question}\n\n{alt_text}\n\n{instruction}"
        claude_content.append({"type": "text", "text": full_prompt})

        # DEBUG PRINT:
        print("\n--- Prompt Sent to Claude ---")
        print(json.dumps(claude_content, indent=2, ensure_ascii=False))
        print("--- End Prompt ---\n")

        try:
            response = client.messages.create(
                model=MODEL,
                temperature=temp,
                max_tokens=1,
                messages=[
                    {
                        "role": "user",
                        "content": claude_content
                    }
                ]
            )
            if not response.content or not response.content[0].text:
                raise ValueError("Empty or malformed response from Claude")
            answer = response.content[0].text.strip().upper()
            if answer not in ["A", "B", "C", "D", "E"]:
                raise ValueError(f"Unexpected answer format: {answer}")
        except Exception as e:
            print(f"❌ Q{q_num}: Failed — {e}")
            answer = "Erro"

        run_answers.append(answer)
        print(f"✅ Q{q_num}: {answer}")
        time.sleep(0.6)

    matrix.iloc[3:, run] = run_answers  # Offset rows by 3 for metadata

    if run % 5 == 0:
        matrix.to_excel(OUTPUT_FILE, index=False)
        print(f"📏 Progress saved after Run {run}")

matrix.to_excel(OUTPUT_FILE, index=False)
print(f"\n✅ All done. Results saved to {OUTPUT_FILE}")
