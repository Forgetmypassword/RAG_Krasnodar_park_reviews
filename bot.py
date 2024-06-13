import os
from dotenv import load_dotenv
import logging
from telegram.ext import (
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    AIORateLimiter,
    filters
)
from telegram.constants import ParseMode
from telegram import Update
from llama_rag import answer_question

# load environment variables
load_dotenv()

# setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# for further possible files handling. path = os.environ.get('PROJECT_PATH', '..')

# start function with greetings and short instruction
async def start(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.effective_chat.id, text='Привет! Я видел много отзывов о Краснодарском парке 🌳\n'
                                                                          'Могу и тебе рассказать. Спрашивай, я не занят.\n',
                                   parse_mode=ParseMode.HTML)


# Handling messages, our main function.
async def message_handle(update: Update, context: CallbackContext):

    _message = update.message.text

    # if message is empty bot asks to provide a message
    if not _message:
        await update.message.reply_text("Пустое сообщение. Попробуй отправить еще раз, пожалуйста.",
                                        parse_mode=ParseMode.HTML)
        return
    # otherwise we try to use our answer_question function from llama_rag.py. 
    # if exception is raised our bot sends a message that it will come back later and logger gives us INFO about an error
    try:
        await update.message.reply_text(answer_question(_message), parse_mode=ParseMode.HTML)
    except Exception as e:
        error_text = f'Something went wrong during prediction. Reason: {e}'
        logger.error(error_text)
        await update.message.reply_text('Произошла ошибка, я посмотрю, что случилось. Попробуй написать мне позже, пожалуйста.',
                                        parse_mode=ParseMode.HTML)

# build the bot app here: run long-polling, provide a token, indicate handlers (start and message)
def run_bot() -> None:
    t_token = os.getenv('TELEGRAM_API_KEY')
    application = (
        ApplicationBuilder()
        .token(t_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .http_version("1.1")
        .get_updates_http_version("1.1")
        .build()
    )

    user_filter = filters.ALL

    application.add_handler(
        CommandHandler("start", lambda update, context: start(update, context), filters=user_filter))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))

    application.run_polling()


if __name__ == '__main__':
    run_bot()