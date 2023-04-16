import vk_api
import bs4
import requests
import urllib.request
import os
import shutil
from tkinter import Tk
import hashlib
from pathlib import Path
from tqdm import tqdm
import json
import asyncio
import schedule
import time
import html
from vk_api.longpoll import VkLongPoll, VkEventType
import vk_captchasolver
from io import BytesIO
from Classifier_train import train_model
from Classifier_evaluate import get_neural_predict
from ChatGPT import Gen_message_by_GPT

with open("token.txt", "r") as f:
    bot_token = f.read()

with open("openAI_token.txt", "r") as f:
    openAI_token = f.read()

def write_msg(user_id, message):
    return vk.method('messages.send', {'random_id': 0, 'user_id': user_id, 'message': message})

def captcha_handler(captcha):
    text = vk_captchasolver.solve(image = BytesIO(requests.get(captcha.get_url()).content))
    #key = ImageToTextTask.ImageToTextTask(anticaptcha_key="4dbd31ff2039565bdb79ffd1b6e60428", save_format='const').captcha_handler(captcha_link=captcha.get_url())

    # Пробуем снова отправить запрос с капчей
    return captcha.try_again(text)

# Авторизуемся как сообщество
vk = vk_api.VkApi(token = bot_token, captcha_handler = captcha_handler)

# Работа с сообщениями
longpoll = VkLongPoll(vk)

Vinchik_ID = "-91050183"
Evil_corp_ID = "-29246653"

min_for_class = 100
max_for_class = 50000

model_girls = "girl_classifier_68.pth"
model_boys = "boy_classifier.pth"

val_k = 5 #коэффициент того, какую долю данных оставить на валидацию

default_settings = {
    "model": model_girls,
    "BL": [],
    "WL": [{"фото": "есть"}],
    "reply": 
    {
        "friends": True,
        "message":
       {
           "friend": True,
           "GPT": True,
           "text": ""
           }
        },
    "stat": True,
    "autolike": 100,
    "autodislike": 100,
    "autopilot": -1,
    "my_form_text": "",
    "is_boy": True,
    "is_friend_boy": False
}

def get_dialogs(vk_local, user_id):
    dialogs = vk_local.messages.getHistory(peer_id = user_id)
    return dialogs['count']

def get_dataset(user_id, vk_local, vlen, elen, user_path, user_settings):
    likes_count = 0
    dislikes_count = 0
    train_path = user_path + "\\train_dir\\"
    path = train_path + "train\\"
    lpath = path + "like\\"
    dpath = path + "dislike\\"
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.mkdir(train_path)
    os.mkdir(path)
    os.mkdir(lpath)
    os.mkdir(dpath)
    two_chats = [[elen, Evil_corp_ID], [vlen, Vinchik_ID]]
    got_self_info = False
    self_info_not_complited = False
    last_message = ""
    my_form_text = ""
    is_boy = None
    for chat in two_chats:
        chat_id = chat[1]
        chat_len = chat[0]
        if chat[0] > 0:
            if chat_id == Evil_corp_ID:
                write_msg(user_id, "Читаю твой чат с Корпорацией зла")
            else:
                write_msg(user_id, "Читаю твой чат с Леонардо Дайвинчиком")
            message_id = write_msg(user_id, f"Сканирую...")
            resid = 0
            activated = False
            off = 0
            while resid < chat_len:
                fh = vk_local.messages.getHistory(peer_id=chat_id, offset = resid, count = 200)
                friend_history = fh['items']
                resid += 200
                for index in friend_history:
                    off += 1
                    if index['from_id'] == user_id and index['text'] != '':
                        last_message = index['text']
                        if index['text'] == '2':
                            is_two = True
                            activated = True
                        elif ord(index['text'][0]) in [10084, 128140] or index['text'] == '1':
                            activated = True
                            is_two = False
                            is_liked = True
                            is_two = False
                        elif index['text'][0] in ['👎', '3']:
                            activated = True
                            is_two = False
                            is_liked = False
                            is_two = False
                        else:
                            activated = False
                            is_two = False
                    elif activated == True:
                        if is_two == True:
                            if index['text'][:31] == "Кому-то понравилась твоя анкета":
                                is_liked = False
                            else:
                                is_liked = True
                        for ind in index['attachments']:
                            if ind['type'] == 'photo':
                                siz = []
                                for iin in ind['photo']['sizes']:
                                    siz.append(iin['height'])
                                maxx = max(siz)
                                for iin in ind['photo']['sizes']:
                                    if iin['height'] == maxx:
                                        imurl = iin['url']
                                        break
                                response = str(requests.head(imurl))
                                if response == '<Response [200]>':
                                    img = urllib.request.urlopen(imurl).read()
                                if is_liked == True:
                                    with open(lpath + str(likes_count) + ".png", "wb") as f:
                                        f.write(img)
                                    likes_count += 1
                                else:
                                    with open(dpath + str(dislikes_count) + ".png", "wb") as f:
                                        f.write(img)
                                    dislikes_count += 1
                        activated = False
                    elif got_self_info == False:
                        if index['text'] in ["Кто тебе интересен?", "Кого тебе найти?\n\n1. Девушку.\n2. Парня.\n3. Все равно."]:
                            if self_info_not_complited == False:
                                l_off = off - 1
                                l_index_from_id = index['from_id']
                                while l_off > -1 and not(l_index_from_id == user_id) and not(last_message in ['1', "Девушки", '2', "Парни", '3', "Не важно"]):
                                    l_off -= 1
                                    local_message = vk_local.messages.getHistory(peer_id=chat_id, offset = l_off, count = 1)
                                    last_message = local_message["text"]
                                    l_index_from_id = local_message["from_id"]
                                if l_off == -1:
                                    self_info_not_complited = True
                                else:
                                    if last_message in ['1', "Девушки"]:
                                        is_friend_boy = False
                                        self_info_not_complited = False
                                    elif last_message in ['2', "Парни"]:
                                        is_friend_boy = True
                                        self_info_not_complited = False
                                    elif last_message in ['3', "Не важно"]:
                                        is_friend_boy = 0
                                        self_info_not_complited = False
                        elif "Расскажи о себе и кого хочешь найти" in index['text']:
                            if last_message == "" or last_message == "Оставить текущий текст" or last_message == '1':
                                self_info_not_complited = True
                            else:
                                my_form_text = last_message
                                self_info_not_complited == False
                        elif index['text'] == "Теперь определимся с полом":
                            if self_info_not_complited == False:
                                l_off = off - 1
                                l_index_from_id = index['from_id']
                                while l_off > -1 and not(l_index_from_id == user_id) and not(last_message in ['1', "Я девушка", '2', "Я парень"]):
                                    l_off -= 1
                                    local_message = vk_local.messages.getHistory(peer_id=chat_id, offset = l_off, count = 1)
                                    last_message = local_message["text"]
                                    l_index_from_id = local_message["from_id"]
                                if l_off != -1:
                                    if last_message in ['1', "Я девушка"]:
                                        is_boy = False
                                        got_self_info = True
                                        self_info_not_complited = False
                                    elif last_message in ['2', "Я парень"]:
                                        is_boy = False
                                        got_self_info = True
                                        self_info_not_complited = False
                vk.method('messages.edit', {'message_id': message_id, 'peer_id': user_id, 'user_id': user_id, 'message': f"Просканировал {min(resid, chat_len)} из {chat_len} сообщений, нашёл {likes_count} лайков и {dislikes_count} дизлайков. Продолжаю сканирование..."})
    user_settings["my_form_text"] = my_form_text
    user_settings["is_boy"] = is_boy
    if is_friend_boy == 0:
        if is_boy == None:
            is_boy = True
        user_settings["is_friend_boy"] = not(is_boy)
    else:
        user_settings["is_friend_boy"] = is_friend_boy
        if is_boy == None:
            is_boy = not(is_friend_boy)
    with open(user_path + '\\settings.json', 'w') as f:
        f.write(json.dumps(user_settings))
    min_class = min(likes_count, dislikes_count)
    if likes_count == 0 and dislikes_count == 0:
        user_settings["model"] = model_girls
        with open(user_path + '\\settings.json', 'w') as f:
            f.write(json.dumps(user_settings))
        return "Сканирование успешно завершено, оценённых анкет не найдено, установлена стандартная модель для оценки девушек"
    if min_class < min_for_class:
        user_settings["model"] = model_girls
        with open(user_path + '\\settings.json', 'w') as f:
            f.write(json.dumps(user_settings))
        return f"У вас недостаточно анкет для начала обучения. Минимально необходимо по {min_for_class} пролайканых анкет и столько же задизлайканых. У вас найдено {likes_count} пролайканых анкет и {dislikes_count} задизлайканых. Установлена стандартная модель для оценки девушек"
    write_msg(user_id, f"Получено {likes_count} пролайканых и {dislikes_count} задизлайканых анкет. Удаляю повторяющиеся анкеты")
    # Удаляем одинаковые файлы
    Tk().withdraw()
    list_of_files = os.walk(path)
    unique_files = dict()
    counter = 0
    for root, _, files in list_of_files:
        for file in tqdm(files):
            file_path = Path(os.path.join(root, file))
            Hash_file = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
            if Hash_file not in unique_files:
                unique_files[Hash_file] = file_path
            else:
                counter += 1
                os.remove(file_path)
    likes_count = sum(1 for _ in Path(lpath).iterdir())
    dislikes_count = sum(1 for _ in Path(dpath).iterdir())
    if likes_count < min_for_class or dislikes_count < min_for_class:
        user_settings["model"] = model_girls
        with open(user_path + '\\settings.json', 'w') as f:
            f.write(json.dumps(user_settings))
        return f"К сожалению, у вас недостаточно оригинальных анкет для начала обучения. Минимально необходимо по {min_for_class} пролайканых анкет и столько же задизлайканых. У вас найдено оасталось {likes_count} пролайканых анкет и {dislikes_count} задизлайканых после удаления. Установлена стандартная модель для оценки девушек"
    write_msg(user_id, f"Осталось {likes_count} пролайканых и {dislikes_count} задизлайканых анкет после удаления повторяющихся. Этого достаточно для начала обучения.")
    if likes_count < dislikes_count:
        delpath = dpath
        max_c = dislikes_count
    else:
        delpath = lpath
        max_c = likes_count
    os.mkdir(train_path + "val\\")
    os.mkdir(train_path + "val\\like\\")
    os.mkdir(train_path + "val\\dislike\\")
    write_msg(user_id, f"Удаляю лишнее, чтобы сравнять классы (для лучшего обучения)")
    j = max_c - 1
    del_count = max_c - min_class
    f_name = delpath + str(j) + ".png"
    for _ in range(del_count):
        while os.path.isfile(f_name) == False:
            j -= 1
            f_name = delpath + str(j) + ".png"
        os.remove(delpath + str(j) + ".png")
    #Делю набор данных на валидационую и тренеровочную
    move_count = min_class // val_k
    list_of_files = os.walk(lpath)
    counter = 0
    for root, _, files in list_of_files:
        for file in tqdm(files):
            if counter < move_count:
                file_path = Path(os.path.join(root, file))
                new_file_path = Path(os.path.join(train_path + "val\\like\\", file))
                shutil.move(file_path, new_file_path)
                counter += 1
            else:
                break
    list_of_files = os.walk(dpath)
    counter = 0
    for root, _, files in list_of_files:
        for file in tqdm(files):
            if counter < move_count:
                file_path = Path(os.path.join(root, file))
                new_file_path = Path(os.path.join(train_path + "val\\dislike\\", file))
                shutil.move(file_path, new_file_path)
                counter += 1
            else:
                break
    write_msg(user_id, f"Получено {min_class} понравившихся анкет и {min_class} отклонённых анкет, начинаю обучение, это может занять до часа в зависимости от загруженности сервера")
    message_id = write_msg(user_id, "Обучаюсь...")
    best_acc = train_model(user_path, message_id, user_id, vk)
    with open(user_path + "\\status.txt", "w") as f:
        f.write('1')
    best_acc *= 100
    user_settings["model"] = user_path + "\\classifier.pth"
    with open(user_path + '\\settings.json', 'w') as f:
        f.write(json.dumps(user_settings))
    if best_acc < 55:
        return f"Ваша модель успешно обучена, однако её точность -- {best_acc:.2f}%, а это практически случайный выбор. Рекомендуется использование стандартной модели. Для повышения точности после обучения на вашей модели рекомендуется оценить больше анкет. Используйте более конкретные критерии оценки. При необходимости, чтобы некоторые анкеты не попадали в набор данных при обучении, удалите их из чата с Винчиком"
    else:
        return f"Ваша модель успешно обучена. Получена точность: {best_acc:.2f}%. Если точность около 50% рекомендуется использовать стандартную модель.\nТеперь, пожалуйста, настройте поведение болта при помощи существующих команд. Для получения справки по командам используйте команду \"!помощь\". Так для автоматической работы бота в полностью автономном режиме вам необходимо сначала активировать \"автопилот\". Для активации бота после установки настроек используется команда \"!старт\""

def scan_messages_from_vinchik(longpoll_user, vk_user):
    for event in longpoll_user.listen():
        if event.type == VkEventType.MESSAGE_NEW:
            if event.to_me:
                if event.peer_id == -91050183:
                    print('Сообщение от Винчика:')
                    print(event.text)
                    message = vk_user.method('messages.getHistory', {'peer_id': -91050183, 'offset': 0, 'count': 1})["items"][0]
                    return message

def send_message_by_user(vk_user, chat_id, message):
    return vk_user.method('messages.send', {'random_id': 0, 'user_id': chat_id, 'message': message})

def operator_check(need_param, got_value):
    if need_param[1] == '=':
        need_value = int(need_param[2:])
        if need_param[0] == '=':
            return True if got_value == need_value else False
        elif need_param[0] == '!':
            return True if got_value != need_value else False
        elif need_param[0] == '>':
            return True if got_value >= need_value else False
        elif need_param[0] == '<':
            return True if got_value <= need_value else False
    else:
        need_value = int(need_param[1:])
        if need_param[0] == '>':
            return True if got_value > need_value else False
        elif need_param[0] == '<':
            return True if got_value < need_value else False

def check_black_white_lists(is_black_list, list_params, message):
    if list_params == []:
        return True
    else:
        pos = message["text"].find("\n")
        if pos == -1:
            description = ""
            form_desc = message["text"]
        else:
            form_desc = message["text"][:message["text"].find("\n")]
            description = message["text"][message["text"].find("\n") + 1:]
        photo = message["photo"]
        pos = form_desc.rfind(",")
        distance = form_desc[pos + 2:]
        form_desc = form_desc[:pos]
        city = ""
        radius = -1
        if distance[0] == '📍':
            pos = distance.find(' ')
            if pos == -1:
                radius = int(distance[1:distance.find('К')]) * 1000
            else:
                radius = int(distance[1:pos])
        else:
            city = distance
        pos = form_desc.rfind(",")
        age = int(form_desc[pos + 2:])
        for param in list_params:
            if "возраст" in param:
                if operator_check(param["возраст"], age) == is_black_list:
                    return False
            elif "радиус" in param:
                if radius != -1 and operator_check(param["радиус"], radius) == is_black_list:
                    return False
            elif "город" in param:
                if city != "":
                    if param["город"][1] == '!':
                        if param["город"][2:-1].upper() != city.upper() == is_black_list:
                            return False
                    else:
                        if param["город"][1:-1].upper() == city.upper() == is_black_list:
                            return False
            elif "фото" in param:
                if param["фото"] == "есть":
                    if photo != None == is_black_list:
                        return False
                else:
                    if photo == None == is_black_list:
                        return False
            elif "описание" in param:
                if param["фото"] == "есть":
                    if description != "" == is_black_list:
                        return False
                else:
                    if description == "" == is_black_list:
                        return False
            else:
                if param[0] == '!':
                    if param[2:-1] in message["text"] == is_black_list:
                        return False
                else:
                    if param[1:-1] in message["text"] == is_black_list:
                        return False

def check_all_cond(message, user_settings):
    is_photo_in_form = False
    if "attachments" in message:
        for attachment in message["attachments"]:
            if attachment["type"] == "photo":
                is_photo_in_form = True
                siz = []
                for iin in attachment['photo']['sizes']:
                    siz.append(iin['height'])
                maxx = max(siz)
                for iin in attachment['photo']['sizes']:
                    if iin['height'] == maxx:
                        imurl = iin['url']
                        break
                response = str(requests.head(imurl))
                if response == '<Response [200]>':
                    photo = urllib.request.urlopen(imurl).read()
                    message["photo"] = photo
    if is_photo_in_form == False:
        message["photo"] = None
    if check_black_white_lists(True, user_settings["BL"], message) == False: #Проверяем чёрный список
        return False, -1
    if check_black_white_lists(False, user_settings["WL"], message) == False: #Проверяем белый список
        return False, -1
    if user_settings["autolike"] == 100 and user_settings["autodislike"] == 100 and user_settings["autopilot"] == -1 or not(is_photo_in_form):
        return True, -1
    else:
        prob = get_neural_predict(user_settings["model"], photo)
        if user_settings["autopilot"] != -1:
            return (False, prob) if prob < user_settings["autopilot"] else (True, prob)
        else:
            if user_settings["autolike"] != 100 and user_settings["autolike"] < prob:
                return True, prob
            if user_settings["autodislike"] != 100 and user_settings["autodislike"] < prob:
                return False, prob
            else: 
                return True, prob

def work_in_Vinchik(user_id, chat_id): #Функция, активирующая сеанс автопилота в беседе с ботом Винчика
    likes_count = 0
    dislikes_count = 0
    pr = 0
    user_path = "Forms\\" + str(user_id)
    with open(user_path + '\\settings.json', 'r', encoding='utf-8') as f:
        user_settings = json.load(f)
    with open(user_path + '\\token.txt', 'r') as f:
        token = f.read()
    vk_user = vk_api.VkApi(token = token, captcha_handler = captcha_handler)
    longpoll_user = VkLongPoll(vk_user)
    message = vk_user.method('messages.getHistory', {'peer_id': chat_id, 'offset': 0, 'count': 1})["items"][0]
    orig_m = html.unescape(message["text"])
    message["text"] = orig_m.upper()
    if message["text"][:24] == "СЛИШКОМ МНОГО ЛАЙКОВ ЗА ":
        send_message_by_user(vk_user, chat_id, "1")
        message = scan_messages_from_vinchik(longpoll_user, vk_user)
        orig_m = html.unescape(message["text"])
        message["text"] = orig_m.upper()
    pos = message["text"].find("\n")
    if pos == -1:
        first_string = message["text"]
    else:
        first_string = message["text"][:pos]
    did_reply = False
    while message["text"][:24] != "СЛИШКОМ МНОГО ЛАЙКОВ ЗА ":
        d_num = '3'
        while first_string.count(",") != 2:
            offset = 0
            pos = message["text"].find("\n")
            if pos == -1:
                first_string = message["text"]
                another_user_form_text = ""
            else:
                first_string = message["text"][:pos]
                another_user_form_text = orig_m[pos + 1:]
                if first_string == "1. СМОТРЕТЬ АНКЕТЫ.":
                    offset = 1
                    message = vk_user.method('messages.getHistory', {'peer_id': chat_id, 'offset': offset, 'count': 1})["items"][0]
                    orig_m = html.unescape(message["text"])
                    message["text"] = orig_m.upper()
                    pos = message["text"].find("\n")
                    first_string = message["text"][:pos]
                    another_user_form_text = orig_m[pos + 1:]
                if "КОМУ-ТО ПОНРАВИЛАСЬ ТВОЯ АНКЕТА" in first_string:
                    d_num = '2'
                    message["text"] = message["text"][pos + 2:]
                    first_string = ",,"
                elif first_string in ["ВРЕМЯ ПРОСМОТРА АНКЕТЫ ИСТЕКЛО, ДЕЙСТВИЕ НЕ ВЫПОЛНЕНО.", "НАШЛИ КОЕ-КОГО ДЛЯ ТЕБЯ ;) ЗАКАНЧИВАЙ С ВОПРОСОМ ВЫШЕ И УВИДИШЬ КТО ЭТО"]:
                    message["text"] = message["text"][pos + 2:]
                    first_string = ",,"
                while first_string[:53] == "ЕСТЬ ВЗАИМНАЯ СИМПАТИЯ! ДОБАВЛЯЙ В ДРУЗЬЯ - VK.COM/ID":
                    offset += 1
                    message = vk_user.method('messages.getHistory', {'peer_id': chat_id, 'offset': offset, 'count': 1})["items"][0]
                    orig_m = html.unescape(message["text"])
                    message["text"] = orig_m.upper()
                    pos = message["text"].find("\n")
                    if pos == -1:
                        first_string = message["text"]
                    else:
                        first_string = message["text"][:pos]
                did_reply = False
                while offset > 1:
                    did_reply = True
                    offset -= 1
                    message = vk_user.method('messages.getHistory', {'peer_id': chat_id, 'offset': offset, 'count': 1})["items"][0]
                    pos = message["text"].find("\n")
                    friend_id = int(message["text"][:pos][53:])
                    message["text"] = message["text"][pos + 2:]
                    orig_m = html.unescape(message["text"])
                    message["text"] = orig_m.upper()
                    pos = message["text"].find("\n")
                    if pos == -1:
                        first_string = message["text"]
                    else:
                        first_string = message["text"][:pos]
                    another_user_form_text = orig_m[pos + 1:]
                    if user_settings["reply"]["message"]["friend"] == True:
                        if user_settings["reply"]["message"]["GPT"] == True:
                            message_to_another_user = Gen_message_by_GPT(token = openAI_token, to_friends = True, my_form_text = user_settings["my_form_text"], another_user_form_text = another_user_form_text, is_boy = user_settings["is_boy"], is_friend_boy = user_settings["is_friend_boy"])
                            try:
                                vk_user.method('friends.add', {'user_id': friend_id, 'text': message_to_another_user})
                            except:
                                try:
                                    vk_user.method('messages.send', {'random_id': 0, 'user_id': friend_id, 'message': message_to_another_user})
                                    write_msg(user_id, f"Не удалось отправить заявку в друзья пользователю vk.com/id{friend_id}, возможно превышен лимит на сообщения. Сообщение отправлено как сообщение, без заявки в друзья.")
                                except:
                                    write_msg(user_id, f"Не удалось отправить заявку в друзья пользователю vk.com/id{friend_id} по неизвестной причине. Сообщение тоже не удалось отправить. Возможно у пользователя закрыты сообщения или вы у него в ЧС.")
                        else:
                            if user_settings["reply"]["message"]["text"] == "":
                                try:
                                    vk_user.method('friends.add', {'user_id': friend_id})
                                except:
                                    write_msg(user_id, f"Не удалось отправить заявку в друзья пользователю vk.com/id{friend_id} по неизвестной причине. Возможно вы у него в ЧС.")
                            else:
                                message_to_another_user = user_settings["reply"]["message"]["text"]
                                try:
                                    vk_user.method('friends.add', {'user_id': friend_id, 'text': message_to_another_user})
                                except:
                                    try:
                                        vk_user.method('messages.send', {'random_id': 0, 'user_id': friend_id, 'message': message_to_another_user})
                                        write_msg(user_id, f"Не удалось отправить заявку в друзья пользователю vk.com/id{friend_id}, возможно превышен лимит на сообщения. Сообщение отправлено как сообщение, без заявки в друзья.")
                                    except:
                                        write_msg(user_id, f"Не удалось отправить заявку в друзья пользователю vk.com/id{friend_id} по неизвестной причине. Сообщение тоже не удалось отправить. Возможно у пользователя закрыты сообщения или вы у него в ЧС.")
            if first_string.count(",") != 2 and did_reply == False:
                send_message_by_user(vk_user, chat_id, "1")
                message = scan_messages_from_vinchik(longpoll_user, vk_user)
                orig_m = html.unescape(message["text"])
                message["text"] = orig_m.upper()
                pos = message["text"].find("\n")
                if pos == -1:
                    first_string = message["text"]
                else:
                    first_string = message["text"][:pos]
        cond, p = check_all_cond(message, user_settings)
        if p != -1:
            pr += p
        if cond == True:
            send_message = "1"
            likes_count += 1
        else:
            send_message = d_num
            dislikes_count += 1
        send_message_by_user(vk_user, chat_id, send_message)
        message = scan_messages_from_vinchik(longpoll_user, vk_user)
        orig_m = html.unescape(message["text"])
        message["text"] = orig_m.upper()
        pos = message["text"].find("\n")
        if pos == -1:
            first_string = message["text"]
        else:
            first_string = message["text"][:pos]
    if user_settings["stat"] == True:
        if likes_count == 0 and dislikes_count == 0:
            return "Слишком много лайков. Попробуй ещё раз через 12 часов"
        else:
            if dislikes_count == 0:
                d = 0
            else:
                d = round(likes_count / dislikes_count, 2)
            avg_p = round(pr / (likes_count + dislikes_count), 2)
            if d > 1.2:
                return f"Сеанс автолайков завершён. Поставлено {likes_count} лайков и {dislikes_count} дизлайков\nСоотношение лайков к дизлайкам равно {d}, лайки ставятся слишком часто, рукомендуется уменьшить значение \"автопилота\"\nСредняя привлекательность анкет, по мнению неронки, составила {avg_p}"
            else:
                return str(f"Сеанс автолайков завершён. Поставлено {likes_count} лайков и {dislikes_count} дизлайков\nСоотношение лайков к дизлайкам равно {d}\nСредняя привлекательность анкет, по мнению неронки, составила {avg_p}")

def read_token(user_id, token):
    vk_session = vk_api.VkApi(token = token, captcha_handler=captcha_handler)
    vk_session._auth_token()
    vk_local = vk_session.get_api()
    v_dialog_len = get_dialogs(vk_local, Vinchik_ID)
    e_dialog_len = get_dialogs(vk_local, Evil_corp_ID)
    with open("Forms\\" + str(user_id) + "\\token.txt", "w") as f:
        f.write(token)
    user_path = "Forms\\" + str(user_id)
    with open(user_path + '\\settings.json', 'r', encoding='utf-8') as f:
        user_settings = json.load(f)
    if v_dialog_len == 0 and e_dialog_len == 0:
        user_settings["model"] = model_girls
        with open(user_path + '\\settings.json', 'w') as f:
            f.write(json.dumps(user_settings))
        return f"У вас не найдено диалогов с Леонардо Дайвинчик или с Корпорацией зла. Установлена стандартная модель для оценки девушек. Для обучения нейронки накопите оценённые анкеты в чате с пинчиком и используйте команду \"!обучить\""
    else:
        count = v_dialog_len + e_dialog_len
        if count < min_for_class * 2:
            user_settings["model"] = model_girls
            with open(user_path + '\\settings.json', 'w') as f:
                f.write(json.dumps(user_settings))
            return f"Данных для обучения будет маловато, у вас всего {count} сообщений в переписке с Винчиком... Минимально необходимо хотябы по {min_for_class} лайков и {min_for_class} дизлайков в сумме. Установлена стандартная модель для оценки девушек. Для обучения нейронки накопите оценённые анкеты в чате с пинчиком и используйте команду \"!обучить\""
        else:
            write_msg(user_id, f"У вас найдено {count} сообщений в переписке с Винчиком. Начинаю анализ")
            return get_dataset(user_id, vk_local, v_dialog_len, e_dialog_len, user_path, user_settings)

class VkBot:
    def __init__(self, user_id):
        self._USER_ID = user_id
        self._USERNAME = self._get_user_name_from_vk_id(user_id)

    def _get_user_name_from_vk_id(self, user_id):
        request = requests.get("https://vk.com/id" + str(user_id))
        bs = bs4.BeautifulSoup(request.text, "html.parser")
    
        user_name = self._clean_all_tag_from_str(bs.findAll("title")[0])
    
        return user_name.split()[0]

    def get_token(self, user_id, token):
        token = token[token.find("access_token=") + 13:]
        if "&" in token:
            token = token[:token.find("&")]
        write_msg(user_id, "Токен получен, начинаю анализ. Это может занять до получаса, в зависимости от загруженности бота и количества сообщений в переписке с Винчиком. Бот будет сканировать, пока не прочтёт все сообщения в переписке с Винчиком. Пожалуйста ожидайте и не пишите сообщения в чат, так вы только замедлите процесс анализа")
        return read_token(user_id, token)
     # Метод для очистки от ненужных тэгов
 
    @staticmethod
    def _clean_all_tag_from_str(string_line):
        """
        Очистка строки stringLine от тэгов и их содержимых
        :param string_line: Очищаемая строка
        :return: очищенная строка
        """
        result = ""
        not_skip = True
        for i in list(string_line):
            if not_skip:
                if i == "<":
                    not_skip = False
                else:
                    result += i
            else:
                if i == ">":
                    not_skip = True
    
        return result

    def new_message(self, user_id, message):
        msg = html.unescape(message).upper()
        user_path = "Forms\\" + str(user_id)
        # Начать
        if "НАЧАТЬ" in msg[:7]:
            if os.path.exists(user_path) == False:
                os.mkdir(user_path)
                with open(user_path + "\\status.txt", "w") as f:
                    f.write('0')
                with open(user_path + '\\settings.json', 'w') as f:
                    f.write(json.dumps(default_settings))
                return f"Привет! Для того, чтобы начать, тебе нужно прислать токен от своей страницы. Мы не передаём токены посторонним лицам и используем исключительно для нужд бота. Если беспокоитесь за свою безопастность, то рекомендуем использовать фейковый аккаунт. Вы всегда можете перейти по ссылке повторно, тогда ваш токен обновится и бот потеряет к нему доступ. Для получения токена перейди по ссылке: https://onl.bz/VTTkFYc, затем подождите 5 секунд, нажмите на кнопку, затем нажмите \"разрешить\" и скопируйте ссылку страницы, на которую вас перекинуло и пришлите в чат"
            else:
                with open(user_path + "\\status.txt", "r") as f:
                    status = f.read()
                if status == '0':
                    return f"Привет! Для того, чтобы начать, тебе нужно прислать токен от своей страницы. Мы не передаём токены посторонним лицам и используем исключительно для нужд бота. Если беспокоитесь за свою безопастность, то рекомендуем использовать фейковый аккаунт. Вы всегда можете перейти по ссылке повторно, тогда ваш токен обновится и бот потеряет к нему доступ. Для получения токена перейди по ссылке: https://onl.bz/VTTkFYc, затем подождите 5 секунд, нажмите на кнопку, затем нажмите \"разрешить\" и скопируйте ссылку страницы, на которую вас перекинуло и пришлите в чат"
                else:
                    return "Смотрю ты уже обучал модель. Желаешь воспользоваться старой или уже изменились вкусы?) Если хочешь оставить оставить старую, то просто настрой, как раньше (получить справку по настройкам можно введя !помощь). Если хочешь переобучить -- введи !переобучить"
        # Обучить:
        if "ОБУЧИТЬ" in msg[:8]:
            write_msg(user_id, "Хорошо, сейчас переобучим! Подожди пожалуйста, пока снова соберём данные!")
            with open("Forms\\" + str(user_id) + "\\token.txt", "r") as f:
                token = f.read()
            read_token(user_id, token)

        # Получение токена
        elif "https://oauth.vk.com/blank.html#access_token=vk1.a." in message:
            return self.get_token(user_id, message)
    
        # Помощь
        elif "ПОМОЩЬ" in msg[:7]:
            pos = msg.find(' ')
            if pos > 0:
                arg = msg[pos + 1:]
                if "ПОМОЩЬ" in arg[:7]:
                    return "Команда для получения справки по функционалу бота. Может выполняться как с аргументами, так и без. В качетвте аргументов принимает названия всех остальных команд, таких как \"!Начать\", \"!Помощь\", \"!Обучить\", \"!Старт\", \"!Стоп\", \"!Модель\", \"!ЧС\", \"!БС\", \"!Собщение\", \"!Статистика\", \"!Автолайк\", \"!Автодизлайк\", \"!Автопилот\". При использовании с командой в качестве аргумента выдаёт полную справку по это команде"
                elif "НАЧАТЬ" in arg[:7]:
                    return "Команда начала работы с ботом. Предлагает обучение, если вы обращаетесь к боту впервые, или использовать уже обученную модель. Или, если вы уже обучали модель, предлагает использовать вашу, уже обученную модель"
                elif "ОБУЧИТЬ" in arg[:8]:
                    return "Команда, активирующая обучение бота. Он сибирает анкеты из переписки с Винчиком и Корпорацией зла и начинает обучени. Если вы уже обучали модель, бот переобучит её заново. "
                elif "СТАРТ" in arg[:6]:
                    return "Команда, активирующая работу бота в переписке с Винчиком. Для работы используются настройки по-умолчанию, если не установлены другие при помощи команд \"модель\", \"сообщение\", \"статистика\", \"ЧС\", \"БС\", \"автолайк\", \"автодизлайк\" и \"автопилот\". Для остановки работы бота необходимо прописать команду \"стоп\". Для незамедлительной остановки бота можно сменить ВК токен, перезагрузив страницу с токеном, но это не рекомендуется. В последнем случае при повторной активации бота необходимо будет заново создать токен при помощи бота."
                elif "СТОП" in arg[:5]:
                    return "Команда, останавливающая работу бота в переписке с Винчиком. На данный момент работает не мгновенно, а только после окончания сессии работы бота в паблике с Винчиком, если бот в данный момент лайкает и дизлайкает анкеты. Если вам необходимо остановить бота незамедлительно, что не рекомендуется, можно сменить ВК токен, перезагрузив страницу с токеном. В последнем случае при повторной активации бота необходимо будет заново создать токен при помощи бота."
                elif "МОДЕЛЬ" in arg[:7]:
                    return "Команда для выбора модели. Есть уже обученная на большом наборе данных для оценки анкет девушек. Эта модель установлена по-умолчанию, если нет модели, обученной на твоих вкусах. Если ты обучал модель, то она устанавливается по-умолчанию.\nКоманда на данный момент имеет 2 аргумента:\n\"показать\" -- показывает, какая модель на данный момент установлена\n\"девушки\" -- готовая модель для оценки девушек\n\"моя\" -- модель, обученная на ваших вкусах. Эту модель можно выбрать, только если вы обучали свою модель. Для обучения можно использовать команду \"!обучить\""
                elif "ЧС" in arg[:3]:
                    return "Команда для настройки чёрного списка, т.е. если параметры настройки чёрного списка установлены, бот будет автоматически ставить дизлайк всем анкетам, котрые попадают под условия. Возможные аргументы команды:\n\"показать\" -- показывает текущие настройки вашего чёрного спика. Пример использовангия \"!ЧС показать\"\n\"очистить\" -- полностью очищает настройки вашего чёрного спика. Пример использования \"!ЧС очистить\"\nСледующие аргументы нужно указывать через запятую, если желаете установить несколько параметров. Можно прописывать по отдельности. Пример возможных записей \"!ЧС возраст<=16, радиус>2000, город другой, описание==\"есть парень\"\", \"мне 13\", \"мне 14\". Тоже самое можно вписать и по очереди, аргументы будут суммироваться:\n\"!ЧС возраст<=16\"\n\"!ЧС радиус>2000\"\n\"!ЧС \"есть парень\"\"\n\"возраст\" -- установка ограничения по возрасту: после аргумента необходимо написать \"<\", \"<=\", \"!=\", \"==\", \">=\" или \">\" и после целое число, обозначающее возраст. Пример использования \"!ЧС возраст==16\" -- автоматически дизлайкать всех, чей возраст в анкете равен 16 годам. Аналогично с другими операторами сравнения.\n\"радиус\" -- аргумент, аналогичный возрасту, используется вместе с операторами сравнения \"<\", \"<=\", \"!=\", \"==\", \">=\" или \">\" и после целое число, обозначающее расстояние. Срабатывает только для анкет, в оторых указан радиус, не город! Расстояние указывается в метрах! В 1 километре 1000 метров!\n\"город\" -- аргумент проверяет соответствие города. Используется только для анкет, в которых указан город, не радиус. Существуют три варианта использования \"!ЧС город мой\" -- автоматически дизлайкает все анкеты, в который указан тот же город, что и у тебя. \"!ЧС город другой\" -- автоматически дизлайкает все анкеты, в которых указан город, но не тот, что у тебя. \"!ЧС город==\"Москва\"\" -- если использовать оператор \"==\", то будет автоматичсеки дизлайкать те анкеты, котор в которых соответствует наименованию, указанному после аргумента \"город==\". Также можно использовать оператор \"!=\", тогда будут дизлайкаться все анкеты, не удовлетворяющие условию, написанному после \"!=\"\n\"фото\" -- аргумент, определяющий поведение на наличие или отсутствие фото в анкете. Возможны 2 варианта использования: \"!ЧС фото нет\" -- автоматически дизлайкает анкеты, в которых нет фото. \"!ЧС фото есть\" -- автоматически диздайкает все анкеты, в которых есть фото, не знаю, зачем это может потребоваться...\n\"описание\"  -- аргумент, определяющий поведение на наличие или отсутствие описания в анкете. Возможны 3 варианта использования: \"!ЧС описание нет\" -- автоматически дизлайкает анкеты, в которых нет фото. \"!ЧС описание есть\" -- автоматически диздайкает все анкеты, в которых есть описание. \"!ЧС описание нет\" -- автоматически дизлайкает все анкеты, у которых нет описания. И третий вариант -- использование операторов \"==\" и \"!=\" после аргумента \"описание\". Не забывайте брать текст в ковычки при таком сценарии использования. Пример: \"!ЧС описание==\"есть парень\"\" -- автомачитески будет дизлайкать все анкеты, в описаниях которых будет найден текст, написанный после оператора \"==\" и взятый в кавычки. Аналогично с оператором \"!=\", но наоборот.\nТакже поддерживаются аргументы, просто взятые в кавычки. То есть автоматически будут дизлайкаться анкеты, в тексте которых будет найдена строка, взятая в кавычки. Пример \"!ЧС \"13\"\" -- автоматически будут дизлайкаться все анкеты, во всём тексте которых будет найдена строка \"13\", включая, собственно, возраст. Если хотите, наоборот, чтобы дизлайкались все анкеты, в тексте которых отсутствует определённая строка, то перед ковычками строки поставьте оператор \"!\". Пример: \"!ЧС !\"люблю гулять\"\" -- по сути будет аналогична команде \"!БС \"люблю гулять\"\", то есть будут автоматически дизлайкаться все анкеты, в тексте которых не будет найдено строки \"люблю гулять\""
                elif "БС" in arg[:3]:
                    return "Команда для настройки белого списка. Полностью аналогична команде \"!ЧС\", но работает наоборот. То есть автоматически дизлайкает те анкеты, которые НЕ удовлетворяют указанным условиям. Для справки по аргументам команды вызовите справку по команде \"!ЧС\", аргументы у этой команды аналогичные"
                elif "ОТВЕТ" in arg[:6]:
                    return "Команда, задающая правила для поведения в случае, если лайкнули вас (или ответили на ваш лайк). По-умолчанию установлены параметры \"друзья да, сообщение друг, GPT\". Возможные аргументы:\n\"показать\" -- показывает список установленых в данный момент настроек поведения при лайке вашей анкеты, а также при взаимном лайке.\n\"нет\" -- устанавливает настройки поведения на отсутствие поведения, то есть на \"друзья нет, сообщение нет\". Пример использования \"!Ответ нет\"\n\"друзья\" -- аргумент, отвечающий за то, добавлять ли автоматически в друзья (или подписываться, если возможность добавления в друзья у человека закрыта) лайкнувшего вас человека или нет. Добавление в друзья будет осуществимо только если не исчерпан ваш суточный лимит на добавление в друзья, не забывайте об этом. Предусматривает два варианта использования \"!Ответ друзья нет\" и \"!ответ друзья да\" -- не добавлять в друзья (не подписываться) и добавлять (подписываться) соотвественно.\n\"сообщение\" -- аргумент, отвечающий за автоматическую отправку сообщения, имеет 2 подаргумента, которые можно писать как по отдельности, так и вместе, через запятую. Первый подаргумент -- \"друг\", команда с этим аргументом имеет 2 возможных варианта использования: \"да\" и \"нет\". Пример использования\"!Ответ сообщение друг нет\" -- если аргумент \"друзья\" сообщения команды \"!Ответ\" установлен на \"да\", то сообщение автоматически будет присылаться в соответствии с параметром второго подаргумента аргумента \"сообщение\", а он по-умолчанию установлен на \"GPT\", если не установлено ручного автосообщения. Подаргумент \"друг\" в позиции \"да\" означает только то, что первое сообщение для собеседника будет прислано как приоржение к заявке в друзья, а не как сообщение. Если установлено \"сообщение друг нет\", то сообщение отправится как обычное сообщение. Но учтите, что у ВК есть лимит на количество возможных сообщений, отправленных незнакомцам втечение суток. Поэтому параметры \"сообщение друг да\" позволяют увеличить этот лимит в ущерб заметности сообщения для собеседника. Вторым подаргументом аргумента \"сообщение\" команды \"!ответ\" могут служить либо \"GPT\" (писать без ковычек, пример: \"!Ответ сообщение GPT\") -- это будет означать, что ваше первое сообщение будет сгенерированно посредствам нейронной сети ChatGPT 3.5 на основании вашей анкеты и анкеты вашего собеседника, либо сообщение, написанное вручную. Текст такого сообщения пишется в ковычках и автоматически удаляет аргумент \"сообщение GPT\". Пример использования: \"!Ответ сообщение \"Привет, как дела?\"\""
                elif "СТАТИСТИКА" in arg[:11]:
                    return "Команда, настривающая вывод сообщений о статистике после каждой сессии автопилота. По-умолчанию включена. Принимает в качестве аргументов \"вкл\" и \"выкл\". \"!Статистика выкл\" -- выключает вывод сообщений о статистике, \"!Статистика вкл\" -- включает сообщения о статистике"
                elif "АВТОЛАЙК" in arg[:9]:
                    return "Команда, отвечающая за автоматическую установку лайков анкетам, которые нейронка определила как нравящиеся вам с вероятностью, больше установленной. При использовании режима \"автопилот\" команда полностью игнорируется. По-умолчанию установлено значение 100, т.е. выключен. В качетстве аргументов принимает: \n\"показать\" -- показывает текущий статус настройки.\nА также аргументом может являться любое целое число от 0 до 100 отвечающее за то, при какой определённой вероятности лайка будет ставиться лайк. То есть значение 100 будет означать, что лайк будет ставиться только в том случае, если вероятность лайка будет определена нейросетью как 100 процентов, чего быть не может, а значит при этом значении автолайк будет выключен. Если установить 0, то лайк будет ставиться всем анкетам, что также не имеет смысла. Рекоммендуются значения около 50-ти."
                elif "АВТОДИЗЛАЙК" in arg[:12]:
                    return "Команда, отвечающая за автоматическую установку дизлайков анкетам, которые нейронка определила как ненравящиеся вам с вероятностью, больше установленной. При использовании режима \"автопилот\" команда полностью игнорируется. По-умолчанию установлено значение 100, т.е. выключен. В качетстве аргументов принимает: \n\"показать\" -- показывает текущий статус настройки.\nА также аргументом может являться любое целое число от 0 до 100 отвечающее за то, при какой определённой вероятности лайка будет ставиться лайк. То есть значение 100 будет означать, что лайк будет ставиться только в том случае, если вероятность лайка будет определена нейросетью как 100 процентов, чего быть не может, а значит при этом значении автодизлайк будет выключен. Если установить 0, то дизлайк будет ставиться всем анкетам, что также не имеет смысла. Рекоммендуются значения около 50-ти."
                elif "АВТОПИЛОТ" in arg[:10]:
                    return "Команда, отвечающая за автоматическую оценку анкет каждые 12 часов. По-умолчанию отключена. Для запуска автопилота с установленными настройками необходимо воспользоваться командой \"!Старт\". Для остановки -- команду \"!Стоп\" Принимает следующие аргументы:\n\"показать\" -- показывает текущий статус и установленные настроки команды.\n\"вкл\" -- включить \"автопилот\". А также принимает целое число от 0 до 100 в качестве аргумента, по-умолчанию 50. Это число отвечает за то, при какой вероятности лайка, определённого нейросетью, ставится лайк, а при какой -- дизлайк. Т.е. Если установлена вероятность 50, в случае, если нейросеть определить вероятность лайка < 50 бдет установлен дизлайк, иначе -- лайк. Если вы хотите, чтобы бот ставил лайки чаще -- уменьшите число, иначе -- увеличьте"
                else:
                    return "Аргумент команды не распознан. Используйте \"!Помощь помощь\" для получения справки по команде"
            else:
                return "Список всех существующих на данный момент команд:\n\nНачать — Активация бота\n!помощь — получение как общей справки по всем командам, так и справки по отдельным командам, если ввести её в качестве аргумента\n!обучить — запускает обучение бота на новых анкетах (активируется по-умолчанию при первом использовании бота, после отправки токена). Необходимо использовать если, скажем, изменились вкусы\n!старт — активировать работу бота в беседе с Винчиком\n!стоп — отключить работу бота в паблике с Винчиком\n!модель — позволяет выбрать модель для классификации. Либо уже готовую (распределяет только девушек), либо обученную на ваших анкетах\n!ЧС — если аргументы команды будут найдены в анкете, то ей автоматически будет поставлен дизлайк. Также можно настроить фильтрацию по возрасту, расстоянию и приоритет\n!БС — если аргументы команды не будут найдены в анкете, то ей автоматически будет поставлен дизлайк. Также можно настроить фильтрацию по возрасту, расстоянию и приоритет\n!сообщение — настраивает варианты рассылки первых сообщений в случае взаимного лайка\n!статистика — отправлять статистику по мере работы бота (сколько анкет лайкнул, какая точность и так далее)\n!автолайк — настройка пороговой точности, при которой лайк будет установлен автоматически\n!автодизлайк — настройка пороговой точности, при которой дизлайк будет установлен автоматически\n!автопилот — бот начинает автоматически лайкать и дизлайкать анкеты каждые 12 часов в соответствии с настройками фильтров и выбранной моделью оценки фотографий. Активация этой опции автоматически игнорирует настройки \"!автолайк\" и \"!автодизлайк\"\n\nДля получения более подробной информации по той или иной команде введите команду \"!помощь\" с интересующей вас командой в качестве аргумента. Пример \"!помощь ЧС\" справка по команде \"!ЧС\""
        elif "СТАРТ" in msg[:6]:



            write_msg(user_id, "Бот успешно запущен")
            return work_in_Vinchik(user_id, Vinchik_ID)


        elif "СТОП" in msg[:5]:
            return "Команда стоп ещё не добавлена"
        elif "МОДЕЛЬ" in msg[:7]:
            pos = msg.find(' ')
            if pos > 0:
                with open(user_path + '\\settings.json', 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                args = msg[pos + 1:]
                if "ПОКАЗАТЬ" in args[:9]:
                    if settings["model"] == model_girls:
                        return "Установлена стандартная модель для оценки анкет девушек"
                    elif settings["model"] == model_boys: 
                        return "Установлена стандартная модель для оценки анкет парней"
                    else:
                        return "Установлена обученная на ваших вкусах модель"
                if "ДЕВУШКИ" in args[:8]:
                    settings["model"] = model_girls
                    with open(user_path + '\\settings.json', 'w') as f:
                        f.write(json.dumps(settings))
                    return "Стандартная модель для оценки девушек успешно установлена"
                if "МОЯ" in args[:4]:
                    settings["model"] = user_path + "\\classifier.pth"
                    with open(user_path + '\\settings.json', 'w') as f:
                        f.write(json.dumps(settings))
                    return "Модель, обученная на ваших вкусах успешно установлена"
                else:
                    return "Аргумент команды не распознан. Используйте \"!Помощь модель\" для получения справки по команде"
            else:
                return "Команда должна иметь аргумент. Используйте \"!Помощь модель\" для получения справки по команде"
        elif "ЧС" in msg[:3] or "БС" in msg[:3]:
            if "ЧС" in msg[:3]:
                text = "чёрн"
                text_short = "ЧС"
                d_id = "BL"
            else:
                text = "бел"
                text_short = "БС"
                d_id = "WL"
            pos = msg.find(' ')
            if pos > 0:
                with open(user_path + '\\settings.json', 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                args = msg[pos + 1:]
                while pos > 0:
                    pos = args.find(',')
                    if pos == -1:
                        pos = len(args)
                    arg = args[:pos]
                    if arg == '':
                        break
                    args = args[pos + 1:]
                    if len(args) != 0 and args[0] == ' ':
                        args = args[1:]
                    if "ПОКАЗАТЬ" in arg[:9]:
                        if settings[d_id] == []:
                            return "Ваш " + text + "ый список пуст"
                        else:
                            return str(settings[d_id])
                    elif "ОЧИСТИТЬ" in arg[:9]:
                        settings[d_id] = []
                        with open(user_path + '\\settings.json', 'w') as f:
                            f.write(json.dumps(settings))
                        return "Ваш " + text + "ый список очищен"
                    elif "ВОЗРАСТ" in arg[:8]:
                        settings[d_id].append({"возраст": arg[arg.find("ВОЗРАСТ") + 7:]})
                        with open(user_path + '\\settings.json', 'w') as f:
                            f.write(json.dumps(settings))
                    elif "РАДИУС" in arg[:7]:
                        settings[d_id].append({"радиус": arg[arg.find("РАДИУС") + 6:]})
                        with open(user_path + '\\settings.json', 'w') as f:
                            f.write(json.dumps(settings))
                    elif "ГОРОД" in arg[:6]:
                        settings[d_id].append({"город": arg[arg.find("ГОРОД") + 5:]})
                        with open(user_path + '\\settings.json', 'w') as f:
                            f.write(json.dumps(settings))
                    elif "ФОТО" in arg[:5]:
                        settings[d_id].append({"фото": arg[arg.find("ФОТО") + 4:]})
                        with open(user_path + '\\settings.json', 'w') as f:
                            f.write(json.dumps(settings))
                    elif "ОПИСАНИЕ" in arg[:9]:
                        settings[d_id].append({"описание": arg[arg.find("ОПИСАНИЕ") + 8:]})
                        with open(user_path + '\\settings.json', 'w') as f:
                            f.write(json.dumps(settings))
                    elif arg[0] == '\"':
                        settings[d_id].append(arg)
                        with open(user_path + '\\settings.json', 'w') as f:
                            f.write(json.dumps(settings))
                    else:
                        return f"Аргумент {arg} команды не распознан. Используйте \"!Помощь " + text_short + "\" для получения справки по команде"
                return "Настройки " + text + "ого списка успешно применены"
            else:
                return "Команда должна иметь аргумент. Используйте \"!Помощь " + text_short + "\" для получения справки по команде"
        elif "ОТВЕТ" in msg[:6]:
            pos = msg.find(' ')
            if pos > 0:
                with open(user_path + '\\settings.json', 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                args = msg[pos + 1:]
                while pos > 0:
                    pos = args.find(',')
                    if pos == -1:
                        pos = args.find(' ')
                        if pos == -1:
                            pos = len(args)
                    arg = args[:pos]
                    args = args[pos + 1:]
                    if "ПОКАЗАТЬ" in arg[:9]:
                        return "друзья: " + "да" if settings["reply"]["friends"] else "нет" + ", " + "сообщение: (друг: " + "да" if settings["reply"]["message"]["friend"] else "нет" + ", " + "GPT" if "GPT" in settings["reply"]["message"] else ("нет" if settings["reply"]["message"]["text"] == "" else ("\"" + settings["reply"]["message"]["text"] + "\"")) + ")"
                    elif "НЕТ" in arg[:4]:
                        settings["reply"]["friends"] = "нет"
                        settings["reply"]["message"]["GPT"] = False
                        settings["reply"]["message"]["text"] = ""
                        with open(user_path + '\\settings.json', 'w') as f:
                            f.write(json.dumps(settings))
                        return "Автоответ успешно отключён"
                    elif "ДРУЗЬЯ" in arg[:7]:
                        pos = max(args.find(','), args.find(' '), len(args))
                        arg = args[:pos]
                        args = args[pos + 1:]
                        pos = 0
                        if "ДА" in arg[:3]:
                            settings["reply"]["friends"] = True
                        elif "НЕТ" in arg[:4]:
                            settings["reply"]["friends"] = False
                        else:
                            return f"Некорректное значение подаргумента {arg} аргумента \"друзья\" команды \"!ответ\". Для получения справки по команде используйте \"!помощь ответ\""
                        with open(user_path + '\\settings.json', 'w') as f:
                            f.write(json.dumps(settings))
                    elif "СООБЩЕНИЕ" in arg[:10]:
                        pos = max(args.find(','), args.find(' '), len(args))
                        arg = args[:pos]
                        args = args[pos + 1:]
                        if "GPT" in arg[:4]:
                            settings["reply"]["message"]["GPT"] = True
                        elif "\"" in arg[0]:
                            settings["reply"]["friends"]["GPT"] = False
                            settings["reply"]["friends"]["text"] = arg[1:-1]
                        else:
                            return f"Некорректное значение подаргумента {arg} аргумента \"сообщение\" команды \"!ответ\". Если вы пытаетесь написать текст сообщения, то пишите его в ковычках. Для получения полной справки по команде используйте \"!помощь ответ\""
                        with open(user_path + '\\settings.json', 'w') as f:
                            f.write(json.dumps(settings))
                    else:
                        return f"Аргумент {arg} команды не распознан. Используйте \"!Помощь ответ\" для получения справки по команде"
                return "Настройки автоответчика успешно применены"
            else:
                return "Команда должна иметь аргумент. Используйте \"!Помощь " + text_short + "\" для получения справки по команде"
        elif "СТАТИСТИКА" in msg[:11]:
            pos = msg.find(' ')
            if pos > 0:
                with open(user_path + '\\settings.json', 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                args = msg[pos + 1:]
                if "ПОКАЗАТЬ" in args[:9]:
                    if settings["stat"] == True:
                        return "На данный момент статистика выводится"
                    else: 
                        return "На данный момент статистика выводится"
                if "ДА" in args[:3]:
                    settings["stat"] = True
                    with open(user_path + '\\settings.json', 'w') as f:
                        f.write(json.dumps(settings))
                    return "Вывод статистики успешно активирован"
                if "НЕТ" in args[:4]:
                    settings["stat"] = True
                    with open(user_path + '\\settings.json', 'w') as f:
                        f.write(json.dumps(settings))
                    return "Статистика больше не будет выводиться"
                else:
                    return "Аргумент команды не распознан. Используйте \"!Помощь статистика\" для получения справки по команде"
            else:
                return "Команда должна иметь аргумент. Используйте \"!Помощь статистика\" для получения справки по команде"
        elif "АВТОЛАЙК" in msg[:9]:
            pos = msg.find(' ')
            if pos > 0:
                with open(user_path + '\\settings.json', 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                arg = msg[pos + 1:]
                if "ПОКАЗАТЬ" in arg[:9]:
                    return "Текущий параметр автолайка: " + str(settings["autolike"]) + " %"
                else:
                    try:
                        al = int(arg)
                        settings["autolike"] = al
                        with open(user_path + '\\settings.json', 'w') as f:
                            f.write(json.dumps(settings))
                        return "Параметр автолайка успешно установлен на: " + str(al) + "%"
                    except:
                        return f"Аргумент {arg} команды не распознан. Используйте \"!Помощь автолайк\" для получения справки по команде"
            else:
                return "Команда должна иметь аргумент. Используйте \"!Помощь автолайк\" для получения справки по команде"
        elif "АВТОДИЗЛАЙК" in msg[:12]:
            pos = msg.find(' ')
            if pos > 0:
                with open(user_path + '\\settings.json', 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                arg = msg[pos + 1:]
                if "ПОКАЗАТЬ" in arg[:9]:
                    return "Текущий параметр автодизлайка: " + str(settings["autodislike"]) + " %"
                else:
                    try:
                        dl = int(arg)
                        settings["autodislike"] = dl
                        with open(user_path + '\\settings.json', 'w') as f:
                            f.write(json.dumps(settings))
                        return "Параметр автолайка успешно установлен на: " + str(dl) + "%"
                    except:
                        return f"Аргумент {arg} команды не распознан. Используйте \"!Помощь автодизлайк\" для получения справки по команде"
            else:
                return "Команда должна иметь аргумент. Используйте \"!Помощь автодизлайк\" для получения справки по команде"
        elif "АВТОПИЛОТ" in msg[:10]:
            pos = msg.find(' ')
            if pos > 0:
                with open(user_path + '\\settings.json', 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                arg = msg[pos + 1:]
                if "ПОКАЗАТЬ" in arg[:9]:
                    return "Автопилот: " + "отключён" if settings["autopilot"] == -1 else (str(settings["autopilot"]) + " %")
                elif "ВКЛ" in arg[:4]:
                    settings["autodislike"] = 50
                    with open(user_path + '\\settings.json', 'w') as f:
                        f.write(json.dumps(settings))
                    return "Автопилот включён и установлен на 50%"
                elif "ВЫКЛ" in arg[:5]:
                    settings["autodislike"] = -1
                    with open(user_path + '\\settings.json', 'w') as f:
                        f.write(json.dumps(settings))
                    return "Автопилот отключён"
                else:
                    try:
                        ap = int(arg)
                        settings["autopilot"] = ap
                        with open(user_path + '\\settings.json', 'w') as f:
                            f.write(json.dumps(settings))
                        return "Параметр баланса автопилота успешно установлен на: " + str(ap) + "%"
                    except:
                        return f"Аргумент {arg} команды не распознан. Используйте \"!Помощь автопилот\" для получения справки по команде"
        else:
            return "Некорректная команда. Напишите \"!Помощь\" для получения списка команд"

async def server():
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW:
            if event.to_me:
        
                print('Новое сообщение:')
                print(f'От: {event.user_id}\n', end='')



                bot = VkBot(event.user_id)
                write_msg(event.user_id, bot.new_message(event.user_id, event.text))

                '''
                try:
                    bot = VkBot(event.user_id)
                    write_msg(event.user_id, bot.new_message(event.user_id, event.text))
                except:
                    print("\nНепредвиденная ошибка")
                    write_msg(event.user_id, "Работа бота прервана, возможно из-за капчи. Пожалуйста введите капчу и перезапустите бота командой \"!старт\"")
                '''
                print('Текст: ', event.text)

if __name__ == '__main__':
    print("Сервер запущен")
    start_server = server()
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()