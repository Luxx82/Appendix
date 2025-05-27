# consensus_rerun_final.py – 30 Apr 2025
"""
Pipeline de consenso GPT‑4o ⇆ Claude‑3‑Sonnet‑20250219 (vision).

Fluxo de 3 passos:
1. **Resposta inicial** – LLM‑1 justifica cada alternativa e escolhe a letra.
2. **Revisão**           – LLM‑2 diz se concorda; se não, letra nova + breve razão.
3. **Decisão final**     – LLM‑1 vê sua resposta e a revisão; mantém ou altera.

Regras-chave
------------
* A mensagem do usuário em cada passo contém **imagem (opcional) + texto**.
* Para GPT‑4o: `[ {image_url}, {text} ]`.
* Para Claude: `[ {text:"Imagem:"}, {image_url}, {text} ]` → convertido para
  `{image/source:url}` dentro de `send()`.  
* **Resposta deve começar em linha separada com só a letra (A‑E)**.
* Campo `system` só é enviado se não vazio (evita erro 400 da Anthropic).
* `FORCE_BASE64=True` embute a figura (≤ 5 MB) se o host bloquear hot‑link.
* Salva parciais a cada 10 questões e resultado final em Excel.
"""

import os, re, time, base64, requests
from datetime import datetime
import pandas as pd
import openai, anthropic
from anthropic import RateLimitError, NotFoundError
from dotenv import load_dotenv

# ───────── CONFIG ─────────
GPT_MODEL    = "gpt-4o"
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"   # vision
RUNS         = [(GPT_MODEL, CLAUDE_MODEL), (CLAUDE_MODEL, GPT_MODEL)]
FORCE_BASE64 = False         # True → embute imagem
# ──────────────────────────

load_dotenv()
openai.api_key   = os.getenv("OPENAI_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

df = pd.read_excel("Prova2023vFinalGPT.xlsx", sheet_name="Questões Completas")

# ---------- helpers ----------

def strict_letter(txt: str) -> str:
    m = re.match(r"\s*([A-E])", txt.strip(), re.I)
    return m.group(1).upper() if m else ""

def img_to_b64(url: str):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        mime = "image/" + url.split(".")[-1].split("?")[0].lower()
        return base64.b64encode(r.content).decode(), mime
    except Exception as e:
        print("⚠️  download img:", e)
        return None, None

def multimodal_block(url: str | None, prompt: str, target: str):
    """Montagem multimodal.
    target: 'gpt' ou 'claude'.
    • GPT → [image,text]. Se FORCE_BASE64=True, embute como data URL.
    • Claude → [label,image,text]. Se FORCE_BASE64=True, usa image/base64.
    """
    if not url or url.lower() == "nan":
        return prompt

    # escolhe formato da parte de imagem
    if FORCE_BASE64:
        data, mime = img_to_b64(url)
        if not data:
            return prompt
        if target == 'gpt':
            img_part = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{data}"  # data URL para OpenAI
                },
            }
        else:  # Claude via base64
            img_part = {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": data},
            }
    else:
        img_part = {"type": "image_url", "image_url": {"url": url}}

    if target == 'gpt':
        return [img_part, {"type": "text", "text": prompt}]

    # Claude: label + image + prompt
    return [
        {"type": "text", "text": "Imagem:"},
        img_part,
        {"type": "text", "text": prompt},
    ]
 
def convert_for_claude(content):
    if not isinstance(content, list):
        return content
    out = []
    for p in content:
        if p.get("type") == "image_url":
            out.append({"type":"image","source":{"type":"url","url":p["image_url"]["url"]}})
        else:
            out.append(p)
    return out

# ---------- envio ----------

def send(model: str, messages: list, temp=0, _try=0):
    try:
        if "claude" in model.lower():
            sys_prompt = "\n".join(m["content"] for m in messages if m["role"] == "system").strip()
            cl_msgs = [{"role": m["role"], "content": convert_for_claude(m["content"])}
                       for m in messages if m["role"] != "system"]
            kwargs = {"model": model, "messages": cl_msgs, "max_tokens": 1024, "temperature": temp}
            if sys_prompt:
                kwargs["system"] = sys_prompt
            return anthropic_client.messages.create(**kwargs).content[0].text

        resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temp)
        return resp["choices"][0]["message"]["content"]

    except (RateLimitError, Exception) as e:
        if isinstance(e, NotFoundError) or _try >= 2:
            raise
        time.sleep(4)
        return send(model, messages, temp, _try + 1)

def save_partial(res):
    fn = f"consensus_partial_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    pd.DataFrame(res).to_excel(fn, index=False)
    print("💾", fn)

# ---------- LOOP ----------

results, count = [], 0
for llm1, llm2 in RUNS:
    for _, row in df.iterrows():
        qn   = row["Número da questão"]
        qtxt = row["Comando da questão"]
        alts = "\n".join(f"{l}) {row[f'Alternativa {l}']}" for l in "ABCDE")
        img  = None if pd.isna(row.get("Imagem")) else str(row["Imagem"]).strip() or None
        if img and img.lower() == "nan":
            img = None

        # STEP 1
        instr = "Responda a questão a seguir. A primeira linha de sua resposta deve conter APENAS a letra da justificativa que você considera a mais correta (A-E). Justifique a sua escolha."
        p1 = f"{qtxt}\n\n{alts}\n\n{instr}"
        msgs1 = [
            {"role":"system","content":"Você é especialista em cardiologia."},
            {"role":"user",  "content": multimodal_block(img, p1, 'gpt' if 'gpt' in llm1 else 'claude')},
        ]
        r1 = send(llm1, msgs1); a1 = strict_letter(r1)
        print("✅1", qn, llm1, a1)

        # STEP 2
        instr2 = ("A resposta a esta questão foi gerada por um outro modelo de LLM. Responda da seguinte maneira: Se concorda que a escolha do modelo 1 é a melhor resposta para a questão, apenas diga: concordo. Se discorda, diga: discordo, sugestão de mudança para letra (A-E), seguido de uma justificativa sobre por que a sua escolha é uma melhor alternativa do que a escolhida pelo modelo 1.")
        p2 = (f"QUESTÃO ORIGINAL:\n{qtxt}\n\n{alts}\n\n"
              f"Resposta do Modelo 1:\n{r1}\n\n{instr2}")
        r2 = send(llm2, [{"role":"user","content": multimodal_block(img, p2, 'gpt' if 'gpt' in llm2 else 'claude')}])
        agree = "Sim" if re.search(r"concordo", r2, re.I) else "Não"
        sug   = strict_letter(r2) if agree == "Não" else ""
        print("✅2", qn, llm2, agree, sug)

        # STEP 3
        instr3 = ("Você é o Modelo 1 original e respondeu à seguinte questão. Um segundo modelo de LLM (Modelo 2) revisou a sua resposta.  Considerando a revisão sugerida do Modelo 2, você pode escolher alterar ou manter a sua escolha inicial. A sua resposta deve ser APENAS a letra da resposta que considera mais provável.")
        p3 = f"""QUESTÃO ORIGINAL:
{qtxt}

{alts}

Resposta inicial do Modelo 1:
{r1}

Revisão do Modelo 2:
{r2}

{instr3}"""
        msgs3 = msgs1 + [
            {"role":"assistant","content": r1},
            {"role":"user",     "content": multimodal_block(img, p3, 'gpt' if 'gpt' in llm1 else 'claude')},
        ]
        r3 = send(llm1, msgs3); a3 = strict_letter(r3)
        print("✅3", qn, llm1, a3)

        results.append({
            "Q#": qn,
            "LLM1": llm1,"LLM2": llm2,
            "Init": a1,"Init Justif": r1,
            "Peer Agree": agree,"Peer Suggest": sug,"Peer": r2,
            "Final": a3,"Final Justif": r3,
        })

        count += 1
        if count % 10 == 0:
            save_partial(results)
        time.sleep(1)

pd.DataFrame(results).to_excel("consensus_results_final.xlsx", index=False)
print("🎉 concluído")