
from fastapi import FastAPI
from utils import download_csv_and_convert, load_to_faiss

app = FastAPI()

@app.get("/update")
def update_data():
    txt = download_csv_and_convert()
    load_to_faiss(txt)
    return {"message": "Dados atualizados com sucesso!"}
