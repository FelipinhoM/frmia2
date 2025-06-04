import os
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

# 🔑 Carrega variáveis de ambiente
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
openai_key = os.getenv("OPENAI_API_KEY")

# 🧠 Caminho do índice vetorizado
faiss_path = "/mnt/data/faiss_index"

# ✅ Verifica se os arquivos do índice FAISS existem
if not os.path.exists(f"{faiss_path}/index.faiss") or not os.path.exists(f"{faiss_path}/index.pkl"):
    print("❌ O índice vetorizado não foi encontrado. Acesse /update primeiro.")
    exit(1)

# 📦 Carrega o índice
db = FAISS.load_local(
    faiss_path,
    OpenAIEmbeddings(openai_api_key=openai_key),
    allow_dangerous_deserialization=True
)

# 🔁 Memória da conversa (simples)
chat_history = {}

# 🤖 Setup do LLM com memória
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0, openai_api_key=openai_key),
    retriever=db.as_retriever()
)

# 💬 Função para responder mensagens
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_input = update.message.text

    # Mantém histórico por usuário
    if user_id not in chat_history:
        chat_history[user_id] = []

    result = qa_chain.invoke({
        "question": user_input,
        "chat_history": chat_history[user_id]
    })

    chat_history[user_id].append((user_input, result["answer"]))

    if "I don't know" in result["answer"] or "não sei" in result["answer"]:
        await update.message.reply_text("🤔 Não encontrei essa informação no histórico dos dados.")
    else:
        await update.message.reply_text(result["answer"])

# 🚀 Inicializa o bot
if __name__ == "__main__":
    app = ApplicationBuilder().token(telegram_token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Bot iniciado com sucesso.")
    app.run_polling()
