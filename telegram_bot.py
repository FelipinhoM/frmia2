import os
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

openai_key = os.getenv("OPENAI_API_KEY")
bot_token = os.getenv("BOT_TOKEN")

# Verifica se índice existe
index_path = "/mnt/data/faiss_index"
index_file = os.path.join(index_path, "index.faiss")

db = None
if os.path.exists(index_file):
    try:
        db = FAISS.load_local(index_path, OpenAIEmbeddings(openai_api_key=openai_key), allow_dangerous_deserialization=True)
        print("✅ Índice FAISS carregado com sucesso.")
    except Exception as e:
        print(f"❌ Erro ao carregar índice: {e}")
else:
    print("⚠️ Índice ainda não vetorizado. Acesse /update para criá-lo.")

async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pergunta = update.message.text

    if db is None:
        await update.message.reply_text("❌ O índice vetorizado ainda não foi criado. Acesse /update no seu outro projeto FastAPI primeiro.")
        return

    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(pergunta)

    if not docs or not docs[0].page_content.strip():
        resposta = "🤔 Nenhum dado encontrado nos registros preenchidos."
    else:
        resposta = docs[0].page_content

    await update.message.reply_text(resposta)

if __name__ == "__main__":
    if not bot_token:
        raise ValueError("❌ BOT_TOKEN não definido como variável de ambiente")

    app = ApplicationBuilder().token(bot_token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))
    print("🤖 Bot do Telegram iniciado.")
    app.run_polling()
