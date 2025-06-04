
import os
import pandas as pd
import requests
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

CSV_URL = os.getenv("CSV_URL", "")

CAMPOS = {
    "Teste": ["Coluna1", "Coluna2"]
}

def download_csv_and_convert():
    r = requests.get(CSV_URL)
    df = pd.read_csv(pd.compat.StringIO(r.text))
    txt_output = ""
    for _, row in df.iterrows():
        tipo = row.get("Tipo_Formulario")
        if tipo not in CAMPOS:
            continue
        base = f"{row['Data']} | {row['Holding']} | {row['Produto']} | {tipo}"
        campos = CAMPOS[tipo]
        dados = [f"{col}: {row.get(col, '')}" for col in campos]
        txt_output += base + "\n" + "\n".join(dados) + "\n" + "-"*40 + "\n"
    with open("/mnt/data/base_final.txt", "w", encoding="utf-8") as f:
        f.write(txt_output)
    return txt_output

def load_to_faiss(txt):
    docs = [Document(page_content=txt)]
    db = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")))
    db.save_local("/mnt/data/faiss_index")
