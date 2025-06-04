
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

telegram_token = os.getenv("TELEGRAM_TOKEN")
openai_key = os.getenv("OPENAI_API_KEY")

db = FAISS.load_local("/mnt/data/faiss_index", OpenAIEmbeddings(openai_api_key=openai_key), allow_dangerous_deserialization=True)
chat_history = []
chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(openai_api_key=openai_key), db.as_retriever())

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    await update.message.reply_text(result["answer"])

if __name__ == "__main__":
    app = ApplicationBuilder().token(telegram_token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()
