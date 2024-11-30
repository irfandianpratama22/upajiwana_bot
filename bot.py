from telegram import Update, InputMediaPhoto
import os
import json
import random
import time
import torch
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from nltk_module import tokenize, bag_of_words, stem
from model import NeuralNet

# Memuat variabel lingkungan
load_dotenv()
TOKEN = os.getenv('BOT_TOKEN_API')
BOT_USERNAME = os.getenv('BOT_USERNAME')

# Menentukan perangkat (device)
device = torch.device('cpu')

# Memuat data intent dan model
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

async def start_command(update: Update, context):
    await update.message.reply_text("Selamat datang di chatbot CV. Upajiwana Kharisma Tour and Travel. Chatbot ini hanya menyediakan informasi umum mengenai layanan yang kami tawarkan. Untuk pemesanan dan pembayaran, Anda dapat menghubungi kami melalui WhatsApp atau langsung mengunjungi kantor kami. Terima kasih 😊")

# Fungsi untuk menyimpan interaksi
def save_interaction(tag, patterns, respons):
    filename = 'interactions.json'
    data = {
        "tag": tag,
        "patterns": patterns,
        "respons": respons
    }
    if os.path.isfile(filename):
        with open(filename, 'r') as file:
            interactions = json.load(file)
    else:
        interactions = []
    interactions.append(data)
    with open(filename, 'w') as file:
        json.dump(interactions, file, indent=4)

# Fungsi untuk menangani respons menggunakan model NLP
def handle_respons(sentence):
    sentence = tokenize(sentence)
    sentence = [stem(word) for word in sentence]
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.76:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if 'respons' in intent and isinstance(intent['respons'], list) and len(intent['respons']) > 0:
                    respons = random.choice(intent['respons'])
                    text = respons if isinstance(respons, str) else "Maaf, saya tidak mengerti."
                    images = []  # Inisialisasi daftar gambar
                    
                    # Memeriksa apakah kunci 'images' ada dan menangani beberapa gambar
                    if 'images' in intent:  # Jika 'images' ada
                        images = intent['images']  # Menyimpan gambar-gambar
                    return {'text': text, 'images': images}  # Mengembalikan respons gambar
                else:
                    return {'text': "Maaf, saya tidak mengerti.", 'images': None}
    else:
        return {'text': "Maaf, saya tidak mengerti.", 'images': None}

# Fungsi untuk menangani pesan masuk
async def handle_message(update: Update, context):
    start_time = time.time()  # Mulai pengukuran waktu
    message = update.message
    user_id = message.chat.id
    text = message.text
    message_type = message.chat.type

    if message_type == 'group' and BOT_USERNAME in text:
        new_text = text.replace(BOT_USERNAME, "").strip()
        respons = handle_respons(new_text)  
        save_interaction(user_id, new_text, respons['text'])
    else:
        respons = handle_respons(text)  
        save_interaction(user_id, text, respons['text'])

    end_time = time.time()  
    response_time = end_time - start_time  # Menghitung waktu respons
    print('Waktu respons:', response_time, 'detik')  # Mencetak waktu respons
    print('Bot:', respons['text'])

    # Mengirimkan teks respons
    await update.message.reply_text(f"{respons['text']}")
    
    # Mengirimkan gambar jika tersedia
    if respons['images']:  # Memeriksa jika ada gambar
        if isinstance(respons['images'], list): 
            media_group = [InputMediaPhoto(image) for image in respons['images']]
            await update.message.reply_media_group(media=media_group)
        else:  # Jika itu adalah gambar tunggal
            await update.message.reply_photo(photo=respons['images'])

# Fungsi utama untuk menjalankan bot
if __name__ == "__main__":
    print('Memulai...')
    app = Application.builder().token(TOKEN).build()

    # Menambahkan handler perintah
    app.add_handler(CommandHandler('start', start_command))

    # Menambahkan handler pesan
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Menjalankan bot
    print('Polling...')
    app.run_polling(poll_interval=3)
