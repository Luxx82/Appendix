import pandas as pd
import time
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# Carrega variáveis de ambiente e inicializa cliente
load_dotenv()
client = OpenAI()

# Configurações
INPUT_FILE = "Prova2023vFinalGPT.xlsx"
OUTPUT_FILE = "o3_single_run_output.xlsx"
MODEL = "o3-2025-04-16"
LOG_FILE = "o3_response_log.txt"
SLEEP_SECONDS = 0.6
# Ajuste: limite de tokens para não exceder contexto
MAX_COMPLETION_TOKENS = 100000  # Razão: mensagens ~150 tokens, total <200k

# Função para extrair texto de resposta multimodal
def extract_text(message):
    content = getattr(message, 'content', None)
    if content:
        return content.strip()
    attachments = getattr(message, 'attachments', None)
    if attachments:
        for att in attachments:
            if att.get('type') == 'text':
                return att.get('text', '').strip()
    return ''

# Limpa log anterior
with open(LOG_FILE, "w", encoding="utf-8") as log_file:
    log_file.write("")

# Carrega planilha
try:
    df = pd.read_excel(INPUT_FILE)
except Exception as e:
    print(f"Erro ao carregar {INPUT_FILE}: {e}")
    exit(1)

# Prepara perguntas
df.columns = df.columns.str.lower().str.strip()
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

# Instrução de sistema para respostas
system_instruction = (
    "Você é um avaliador de prova de cardiologia. "
    "Responda estritamente com apenas uma letra maiúscula: A, B, C, D ou E. "
    "Não explique nada além dessa letra. "
    "Qualquer desvio será considerado erro."
)

for _, row in questions.iterrows():
    q_num = row["número da questão"]
    question = str(row["comando da questão"]).strip()
    tipo = str(row["contém imagem?"]).strip().lower()
    img_url = row.get("image url", None)

    # Monta texto de alternativas
    alts = row[["alternativa a","alternativa b","alternativa c","alternativa d","alternativa e"]].fillna("Alternativa em branco")
    alt_text = (
        f"A: {alts['alternativa a']}\n"
        f"B: {alts['alternativa b']}\n"
        f"C: {alts['alternativa c']}\n"
        f"D: {alts['alternativa d']}\n"
        f"E: {alts['alternativa e']}"
    )

    # Prefixo de contexto
    if tipo == "imagem":
        prefix = "Analise a imagem e responda:"
    elif tipo == "video":
        prefix = "Analise o vídeo e responda:"
    else:
        prefix = "Analise o enunciado e responda:"

    # Monta prompt de usuário
    combined_prompt = f"{prefix}\n\n{question}\n\n{alt_text}"
    if isinstance(img_url, str) and img_url.startswith("http"):
        combined_prompt += f"\n\n[Imagem: {img_url}]"

    # Log do prompt
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(f"Q{q_num} PROMPT: {combined_prompt}\n")

    # Mensagens
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": combined_prompt}
    ]

    # Chama o modelo
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_completion_tokens=MAX_COMPLETION_TOKENS
        )
        # Log full response object para debugging
        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(f"Q{q_num} FULL_RESPONSE: {response}\n")
        # Extrai conteúdo de resposta
        message = response.choices[0].message
        raw = extract_text(message)
        # Extrai letra no início
        match = re.match(r"^[A-E]", raw.strip().upper())
        if match:
            ans = match.group(0)
        else:
            raise ValueError(f"Formato inesperado: {raw}")
    except Exception as err:
        error_msg = str(err)
        ans = "Erro"
        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(f"Q{q_num} ERROR: {error_msg}\n")

    # Salva resposta e loga
    answers.append(ans)
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(f"Q{q_num} RAW: {raw if 'raw' in locals() else ''}\n")
        log_file.write(f"Q{q_num} RESP: {ans}\n")
    print(f"✅ Q{q_num}: {ans}" if ans in ["A","B","C","D","E"] else f"❌ Q{q_num}: {error_msg if 'error_msg' in locals() else ans}")
    time.sleep(SLEEP_SECONDS)

# Salva resultados
try:
    df_out = pd.DataFrame({"Q#": questions["número da questão"], "Resposta O3": answers})
    df_out.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ Todas as respostas salvas em {OUTPUT_FILE}")
except Exception as e:
    print(f"Erro ao salvar saída: {e}")
