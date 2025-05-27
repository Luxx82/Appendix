import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load API key
load_dotenv()
client = OpenAI()

# File paths
matrix_path = "gpt4o_runs_matrix_fresh.xlsx"
logprob_path = "gpt4o_logprobs_fresh.xlsx"
input_path = "Prova2023vFinalGPT.xlsx"

# Load questions
df = pd.read_excel(input_path)
questions = df['Número da questão']

# Load or initialize matrix
if os.path.exists(matrix_path):
    matrix_df = pd.read_excel(matrix_path, header=[0, 1, 2])
    matrix_df = matrix_df.set_index(matrix_df.columns[0])
    matrix_df.index.name = "Número da questão"
else:
    matrix_df = pd.DataFrame(index=questions)
    matrix_df.columns = pd.MultiIndex.from_arrays([[], [], []])

# Load or initialize logprobs
if os.path.exists(logprob_path):
    logprobs_df = pd.read_excel(logprob_path, header=[0, 1, 2])
    logprobs_df = logprobs_df.set_index(logprobs_df.columns[0])
    logprobs_df.index.name = "Número da questão"
else:
    logprobs_df = pd.DataFrame(index=questions)
    logprobs_df.columns = pd.MultiIndex.from_arrays([[], [], []])

# Align index (avoids ValueError mismatch)
matrix_df = matrix_df.reindex(questions)
logprobs_df = logprobs_df.reindex(questions)

# Determine starting run number
start_run = len(matrix_df.columns) + 1

# Temperatures for next 30 runs
temps = [0.0] * 10 + [0.3] * 10 + [0.7] * 10

for i, temp in enumerate(temps):
    run_number = start_run + i
    col_name = f"Run {run_number}"
    model = "gpt-4o"
    modality = "simple"

    print(f"\n🚀 Starting {col_name} — Temp: {temp}")
    run_answers = []
    run_logprobs = []

    for idx, row in df.iterrows():
        q_num = row['Número da questão']
        question = str(row['Comando da questão'])
        tipo = str(row.get('Contém Imagem?', '')).strip().lower()
        image_url = str(row.get('Image URL', '')).strip()

        alts = row[['Alternativa A', 'Alternativa B', 'Alternativa C', 'Alternativa D', 'Alternativa E']]
        alt_text = (
            f"A: {alts['Alternativa A']}\n"
            f"B: {alts['Alternativa B']}\n"
            f"C: {alts['Alternativa C']}\n"
            f"D: {alts['Alternativa D']}\n"
            f"E: {alts['Alternativa E']}"
        )

        if tipo == "imagem":
            prefix = "A seguinte pergunta faz referência a uma imagem. Analise o conteúdo da imagem apresentada e, com base nela, responda à pergunta a seguir, escolhendo apenas a alternativa mais provável entre as fornecidas."
        elif tipo == "video":
            prefix = "A imagem abaixo contém uma sequência de quadros extraídos de um vídeo clínico. Analise a progressão dos eventos representados e, com base nela, responda à pergunta a seguir, escolhendo apenas a alternativa mais provável entre as fornecidas."
        else:
            prefix = "A seguinte pergunta faz parte de uma prova de cardiologia. Responda escolhendo apenas a alternativa mais provável entre as fornecidas."

        instruction = (
            "Sua resposta deve ser apenas uma letra (A, B, C, D ou E), sem explicações, palavras adicionais ou pontuação. "
            "Saída = apenas uma única letra maiúscula. "
            "Se não tiver certeza da resposta, ainda assim escolha a melhor alternativa entre as fornecidas."
        )

        full_prompt = f"{prefix}\n\n{question}\n\n{alt_text}\n\n{instruction}"

        content = []
        if isinstance(image_url, str) and image_url.lower().startswith("http"):
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        content.append({"type": "text", "text": full_prompt})

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                temperature=temp,
                max_tokens=1,
                logprobs=True,
                top_logprobs=5
            )
            answer = response.choices[0].message.content.strip()
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            logprob_str = ", ".join([f"{entry.token}:{entry.logprob:.2f}" for entry in top_logprobs])

        except Exception as e:
            answer = "Erro"
            logprob_str = "Erro"
            print(f"⚠️ Q{q_num}: {e}")

        run_answers.append(answer)
        run_logprobs.append(logprob_str)
        time.sleep(0.3)

    # Add new columns with metadata
    matrix_df[(col_name, model, temp)] = run_answers
    logprobs_df[(col_name, model, temp)] = run_logprobs

# Save updated files
matrix_df.to_excel(matrix_path)
logprobs_df.to_excel(logprob_path)

print("\n✅ 30 new runs added.")
print(f"📊 Updated: {matrix_path}")
print(f"📊 Updated: {logprob_path}")
