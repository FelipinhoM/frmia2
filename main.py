import os
import asyncio
from fastapi import FastAPI
from utils import download_csv_and_convert, load_to_faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Variáveis de ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Controle de estado
indice_pronto = False
db = None
chain = None

app = FastAPI()

@app.get("/update")
def update_data():
    global indice_pronto, db, chain
    text = download_csv_and_convert()
    load_to_faiss(text)
    db = FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)
    chain = load_qa_chain(OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff")
    indice_pronto = True
    return {"message": "Índice criado e bot será iniciado."}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not indice_pronto:
        await update.message.reply_text("⚠️ O índice ainda não está pronto. Acesse /update primeiro.")
        return
    query = update.message.text
    docs = db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    await update.message.reply_text(response)

async def iniciar_bot():
    app_telegram = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app_telegram.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Bot iniciado.")
    await app_telegram.run_polling()

@app.on_event("startup")
async def ao_iniciar():
    asyncio.create_task(iniciar_bot())
