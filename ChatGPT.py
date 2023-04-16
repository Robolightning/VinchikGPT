import openai

def Gen_message_by_GPT(token, to_friends, my_form_text, another_user_form_text, is_boy, is_friend_boy):
    openai.api_key = token
    davinchi = True
    # задаем макс кол-во слов
    max_tokens = 512

    if is_boy == True:
        p1 = "парень"
    else:
        p1 = "девушка"

    if is_friend_boy == True:
        p2 = "парнем"
    else:
        p2 = "девушкой"

    if to_friends == True:
        add_to_prompt = ", попроси, чтобы тебя добавили в друзья"
    else:
        add_to_prompt = ""

    if my_form_text == "":
        add_to_prompt_m = ""
    else:
        add_to_prompt_m = f" О себе можешь рассказать следующее: {my_form_text}."

    if another_user_form_text == "":
        p3 = ""
    else:
        p3 = f", чья анкета выглядит следующим образом: {another_user_form_text}"

    # генерируем ответ
    if davinchi == True:
        completion = openai.Completion.create(
            engine = "text-davinci-003",
            prompt = f"Ты {p1} и знакомишься в приложении для знакомств.{add_to_prompt_m} Тебе нужно познакомиться с {p2}{p3}. Напиши текст своего первого сообщения этому человеку{add_to_prompt}, представляться не стоит, поскольку твоё имя и так написано в профиле. Обращайся на \"ты\"",
            max_tokens = max_tokens,
            temperature = 0.5,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0
        )
        r = completion.choices[0].text[2:]
    else:
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [
                    {"role": "system", "content": f"Ты {p1} и знакомишься в приложении для знакомств.{add_to_prompt_m} Тебе нужно познакомиться с {p2}{p3}"},
                    {"role": "user", "content": f"Напиши текст своего первого сообщения этому человеку{add_to_prompt}, представляться не стоит, поскольку твоё имя и так написано в профиле. Обращайся на \"ты\""},
                ]
        )
        r = ''
        for choice in response.choices:
            r += choice.message.content
    r = r.replace(" :)", ")").replace(":)", ")")
    if r[len(r) - 1] == '.':
        r = r[:-1]
    result = r
    return result