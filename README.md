# Let's see Krasnodar park reviews with Llama3 70B and RAG

![Inner child](assets/inner_child.jpg)

## In order to run this project:

## 0. Git clone this repos in the preferred directory

```bash
$ git clone https://github.com/Forgetmypassword/RAG_Krasnodar_park_reviews.git
$ git clone https://github.com/yandex/geo-reviews-dataset-2023.git
```
Ensure that geo-reviews dataset is in the same directory as our RAG project.

## 1. Create a virtual environment and install the required packages:

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## 2. Create a free Pinecone account and get your API key from [here](https://www.pinecone.io/).
Create pinecone index on their website with 3072 dimension, to simplify name it 'reviews'.

## 3. Create a free Groq account and get API key from [here](https://console.groq.com/keys).

## 4. Create a telegram bot and get its token from [botfather](https://t.me/BotFather)

## 5. Create a `.env` file with the following variables:

```bash
PINECONE_API_KEY='ENTER PINECONE API KEY HERE'
GROQ_API_KEY='ENTER GROQ API KEY HERE'
TELEGRAM_API_KEY='ENTER TELEGRAM API KEY HERE'
```

## 6. Pull ollama phi3 model into your machine. We use it as an embedding model. It is lightweight and works great. In terminal: 

```bash
ollama run phi3:3.8b
```
After running you can just stop it or have fun with phi3 3.8b.

## 7. Run all cells in apply_rag_reviews.ipynb. It will create embeddings in your pinecone vectorstore.

## 8. Run llama_rag.py

## 9. Run bot.py

## 10. Have fun!

P.S.: everything was done using VS code (and in jupyter notebook extension for VS code)
