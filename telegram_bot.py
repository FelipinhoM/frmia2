import os
import time
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# Variáveis de ambiente
openai_key = os.getenv("OPENAI_API_KEY")
telegram_token = os.getenv("TELEGRAM_TOKEN")

# Caminho do índice FAISS
INDEX_DIR = "/mnt/data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")

# Esperar o índice vetorizado ser gerado
print("⏳ Aguardando a criação do índice vetorizado...")
while not os.path.exists(INDEX_FILE):
    time.sleep(2)

print("✅ Índice vetorizado encontrado!")

# Carregar base vetorizada
db = FAISS.load_local(INDEX_DIR, OpenAIEmbeddings(openai_api_key=openai_key), allow_dangerous_deserialization=True)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0.1, openai_api_key=openai_key),
    retriever=db.as_retriever()
)

# Handlers do bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Olá! Pode me perguntar qualquer coisa com base nos dados do formulário.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    result = qa_chain.run(question)

    # Resposta padrão se não achar dados relevantes
    if "não sei" in result.lower() or not result.strip():
        response = "Nenhum registro encontrado com base nessa pergunta. Talvez o campo não tenha sido preenchido."
    else:
        response = result

    await update.message.reply_text(response)

# Iniciar o bot
if __name__ == "__main__":
    app = ApplicationBuilder().token(telegram_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Bot está rodando...")
    app.run_polling()
