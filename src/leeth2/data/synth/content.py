"""Content questions about short texts: "is this post about X", "is this a complaint", "is this a complaint about X".

The routine work a main model delegates -- go through posts, bookmarks, messages, tickets and keep those that ...  -- asks a yes/no
question about the CONTENT of a text, and the answer has to be well calibrated: a blatant match must come back as 0.9, not 0.6, or a
fixed accept bound ("act on >= 0.7") never fires.  The router / safety corpora of the other synth modules do not contain such questions.

Every text carries two attributes, a topic and a tone (complaint / praise / question / ad / news).  Three kinds of criteria are built
from them, so that the model has to read both parts of a compound one and cannot answer from the topic or the tone alone:

    topic      "Это про погоду?"                 true when the topic matches
    tone       "Это жалоба?"                     true when the tone matches
    compound   "Это жалоба на погоду?"           true when BOTH match; the negatives include the same topic with another tone and
                                                 the same tone with another topic (the hard ones)

TOPICS_TRAIN / TEXTS_TRAIN are for training; TOPICS_HELD / TEXTS_HELD are never trained on (new topics, written separately) and are
what ``scripts/leeth_content_bench.py`` style evaluations should use for transfer.  The texts are hand-written (no scraping, no personal
data); the wording of the questions is varied on purpose.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from ...schema import Sample

TONES = ("complaint", "praise", "question", "ad", "news")

TONE_QUESTIONS: Dict[str, Dict[str, List[str]]] = {
    "complaint": {"ru": ["Это жалоба?", "Автор чем-то недоволен и жалуется?", "Пост выражает недовольство?"],
                  "en": ["Is this a complaint?", "Is the author unhappy about something?"]},
    "praise": {"ru": ["Автор хвалит что-то?", "Это положительный отзыв?", "Автор доволен и рекомендует?"],
               "en": ["Is this a positive review?", "Does the author praise something?"]},
    "question": {"ru": ["Автор задаёт вопрос или просит совета?", "Это вопрос?", "Автор просит о помощи?"],
                 "en": ["Is the author asking for help or advice?", "Is this a question?"]},
    "ad": {"ru": ["Это реклама или объявление о продаже?", "Это рекламное сообщение?", "Автор что-то продаёт или рекламирует?"],
           "en": ["Is this an advertisement?", "Is this a sales pitch or a promotion?"]},
    "news": {"ru": ["Это новость или информационное сообщение?", "Это сообщение о событии, а не мнение?"],
             "en": ["Is this a news report or an announcement?", "Does this report an event rather than an opinion?"]},
}

TOPIC_QUESTIONS = {
    "ru": ["Это про {acc}?", "Пост относится к теме «{nom}»?", "Текст касается темы: {nom}?", "Речь идёт про {acc}?"],
    "en": ["Is this about {en}?", "Does this post concern {en}?", "Is the topic {en}?"],
}

COMPOUND_QUESTIONS = {
    "complaint": {"ru": ["Это жалоба на {acc}?", "Автор жалуется на {acc}?"], "en": ["Is this a complaint about {en}?", "Does the author complain about {en}?"]},
    "praise": {"ru": ["Это положительный отзыв про {acc}?", "Автор хвалит {acc}?"], "en": ["Is this praise of {en}?", "Is this a positive review about {en}?"]},
    "question": {"ru": ["Это вопрос про {acc}?", "Автор просит совета про {acc}?"], "en": ["Is the author asking about {en}?", "Is this a question about {en}?"]},
    "ad": {"ru": ["Это реклама, связанная с темой «{nom}»?", "Это объявление или реклама про {acc}?"], "en": ["Is this an advertisement about {en}?", "Is this a promotion related to {en}?"]},
    "news": {"ru": ["Это новость про {acc}?", "Это сообщение о событии в теме «{nom}»?"], "en": ["Is this a news item about {en}?", "Is this an announcement about {en}?"]},
}

# topic id -> names in the forms the questions need: nominative, accusative (for "про ..."), English
TOPICS_TRAIN: Dict[str, Dict[str, str]] = {
    "weather": {"nom": "погода", "acc": "погоду", "en": "the weather"},
    "health": {"nom": "здоровье и медицина", "acc": "здоровье и медицину", "en": "health and medicine"},
    "banking": {"nom": "банковские услуги", "acc": "банковские услуги", "en": "banking"},
    "crypto": {"nom": "криптовалюта", "acc": "криптовалюту", "en": "cryptocurrency"},
    "housing": {"nom": "жильё и недвижимость", "acc": "жильё и недвижимость", "en": "housing and real estate"},
    "travel": {"nom": "путешествия", "acc": "путешествия", "en": "travel"},
    "cinema": {"nom": "кино", "acc": "кино", "en": "movies"},
    "music": {"nom": "музыка", "acc": "музыку", "en": "music"},
    "videogames": {"nom": "видеоигры", "acc": "видеоигры", "en": "video games"},
    "books": {"nom": "книги", "acc": "книги", "en": "books"},
    "education": {"nom": "учёба в университете", "acc": "учёбу в университете", "en": "university studies"},
    "pets": {"nom": "домашние животные", "acc": "домашних животных", "en": "pets"},
    "cars": {"nom": "автомобили", "acc": "автомобили", "en": "cars"},
    "fashion": {"nom": "мода и одежда", "acc": "моду и одежду", "en": "fashion and clothing"},
    "fitness": {"nom": "фитнес и тренировки", "acc": "фитнес и тренировки", "en": "fitness"},
    "basketball": {"nom": "баскетбол", "acc": "баскетбол", "en": "basketball"},
    "psychology": {"nom": "психология", "acc": "психологию", "en": "psychology"},
    "space": {"nom": "космос", "acc": "космос", "en": "space exploration"},
    "programming": {"nom": "программирование", "acc": "программирование", "en": "programming"},
    "security": {"nom": "кибербезопасность", "acc": "кибербезопасность", "en": "cybersecurity"},
    "ai": {"nom": "искусственный интеллект", "acc": "искусственный интеллект", "en": "artificial intelligence"},
    "restaurants": {"nom": "рестораны и кафе", "acc": "рестораны и кафе", "en": "restaurants"},
    "telecom": {"nom": "мобильная связь и интернет", "acc": "мобильную связь и интернет", "en": "mobile and internet providers"},
    "taxes": {"nom": "налоги и юридические вопросы", "acc": "налоги и юридические вопросы", "en": "taxes and legal matters"},
    "parenting": {"nom": "воспитание детей", "acc": "воспитание детей", "en": "parenting"},
    "gardening": {"nom": "садоводство", "acc": "садоводство", "en": "gardening"},
    "renovation": {"nom": "ремонт квартиры", "acc": "ремонт квартиры", "en": "home renovation"},
    "utilities": {"nom": "коммунальные услуги", "acc": "коммунальные услуги", "en": "utilities and bills"},
    "transport": {"nom": "общественный транспорт", "acc": "общественный транспорт", "en": "public transport"},
    "shopping": {"nom": "покупки в интернете", "acc": "покупки в интернете", "en": "online shopping"},
    "support": {"nom": "служба поддержки", "acc": "службу поддержки", "en": "customer support"},
    "photography": {"nom": "фотография", "acc": "фотографию", "en": "photography"},
}

TOPICS_HELD: Dict[str, Dict[str, str]] = {
    "delivery": {"nom": "доставка", "acc": "доставку", "en": "delivery"},
    "recipes": {"nom": "кулинария и рецепты", "acc": "кулинарию и рецепты", "en": "cooking and recipes"},
    "football": {"nom": "футбол", "acc": "футбол", "en": "football"},
    "gadgets": {"nom": "гаджеты и техника", "acc": "гаджеты и технику", "en": "gadgets"},
    "politics": {"nom": "политика", "acc": "политику", "en": "politics"},
    "jobs": {"nom": "работа и вакансии", "acc": "работу и вакансии", "en": "jobs and vacancies"},
}

# topic -> tone -> two texts (RU mostly, some EN)
TEXTS_TRAIN: Dict[str, Dict[str, List[str]]] = {
    "weather": {
        "complaint": ["Синоптики опять обещали солнце, а я вторую неделю хожу под дождём без зонта. Прогнозы вообще никуда не годятся.",
                      "Third day of freezing fog and the forecast app still says it is sunny. Absolutely useless."],
        "praise": ["Наконец-то нормальный прогноз: приложение предупредило о грозе за два часа, успели убрать вещи с балкона.",
                   "Love the new weather widget, the hourly rain map is spot on."],
        "question": ["Подскажите, какая погода обычно бывает в Сочи в начале октября, брать ли тёплую куртку?",
                     "Does anyone know why the temperature drops so much after sunset in the desert?"],
        "ad": ["Метеостанция для дачи со скидкой 30%: температура, влажность, давление, доставка по всей стране.",
               "Buy our pocket barometer today and never get caught in the rain again!"],
        "news": ["На выходных в центральных областях ожидается сильный ветер и мокрый снег, службы переведены на усиленный режим.",
                 "A cold front will bring heavy snowfall to the northern regions on Friday, the weather service said."],
    },
    "health": {
        "complaint": ["В поликлинике к терапевту записываться за три недели, а в регистратуре ещё и нагрубили. Как так можно?",
                      "The hospital kept me waiting for six hours in the emergency room and nobody explained anything."],
        "praise": ["Спасибо врачу-кардиологу: принял без очереди, всё объяснил и назначил понятное лечение. Рекомендую.",
                   "Great experience at the new dental clinic, painless and very professional."],
        "question": ["Посоветуйте, к какому специалисту идти, если постоянно болит поясница после работы за компьютером?",
                     "Is it normal to feel tired for a few days after the flu vaccine?"],
        "ad": ["Комплексный медосмотр за один день — скидка 20% до конца месяца. Запись по телефону.",
               "Private clinic offers full body check-up packages, book online and save 15%."],
        "news": ["Минздрав объявил о старте программы бесплатной диспансеризации для людей старше сорока лет.",
                 "Health officials announced a new vaccination campaign starting next week."],
    },
    "banking": {
        "complaint": ["Банк заблокировал карту без объяснений и вторые сутки не может сказать, когда вернёт доступ к деньгам.",
                      "They charged me a monthly fee for a card I closed two months ago. Terrible bank."],
        "praise": ["Очень доволен новым кредитным тарифом: процент ниже, чем у конкурентов, оформили за десять минут.",
                   "The mobile banking app is fast and the support chat answers within a minute."],
        "question": ["Как лучше погасить ипотеку досрочно: уменьшать срок или платёж? Кто сталкивался, поделитесь опытом.",
                     "Can I open a savings account in another country without visiting the branch?"],
        "ad": ["Откройте вклад под 16% годовых до конца месяца — приведи друга и получи бонус 2000 рублей.",
               "Apply for our travel credit card: no annual fee and 5% cashback on flights."],
        "news": ["Центральный банк сохранил ключевую ставку на прежнем уровне и сообщил о планах пересмотра нормативов.",
                 "The bank announced it would close forty branches and move services online by the end of the year."],
    },
    "crypto": {
        "complaint": ["Биржа заморозила вывод средств под предлогом проверки, и уже неделя, как ни ответа, ни денег. Мошенники.",
                      "Lost money because the exchange went down exactly during the price spike. Unacceptable."],
        "praise": ["Перешёл на аппаратный кошелёк и спокойно сплю: удобный интерфейс, быстрые переводы, всё прозрачно.",
                   "Really happy with this staking platform, payouts arrive on time every week."],
        "question": ["Какую комиссию лучше выбирать при переводе биткоина, чтобы транзакция прошла за час?",
                     "Is it safe to keep stablecoins on an exchange for a couple of months?"],
        "ad": ["Курс по трейдингу криптовалют: научим зарабатывать на росте и падении рынка. Первое занятие бесплатно.",
               "Join our crypto signals channel and get daily trading tips, limited places."],
        "news": ["Курс биткоина обновил максимум за последние полгода после решения регулятора разрешить новые фонды.",
                 "Regulators approved the first spot ether fund, trading starts on Thursday."],
    },
    "housing": {
        "complaint": ["Арендодатель поднял цену на треть без предупреждения и ещё удерживает залог за царапину на стене.",
                      "The landlord ignores our messages about the broken heating for three weeks now."],
        "praise": ["Очень довольны новостройкой: хорошая планировка, тихие соседи, застройщик сдал дом раньше срока.",
                   "Found a lovely apartment through this agency, the agent was honest and quick."],
        "question": ["Подскажите, на что обращать внимание при покупке квартиры на вторичном рынке, чтобы не нарваться на долги?",
                     "How much deposit is normal when renting a flat in a big city?"],
        "ad": ["Сдаю двухкомнатную квартиру у метро, свежий ремонт, мебель и техника, без посредников.",
               "New townhouses for sale with a garden, mortgage available from 5%."],
        "news": ["Средняя стоимость квадратного метра в новостройках выросла на 4% за квартал, сообщили аналитики.",
                 "City council approved a plan to build five thousand affordable apartments by 2030."],
    },
    "travel": {
        "complaint": ["Авиакомпания потеряла мой чемодан и уже пять дней не может найти, а в отеле меня заселили в холодный номер без окна.",
                      "Our tour operator cancelled the trip two days before departure and refuses a refund."],
        "praise": ["Поездка на Байкал превзошла ожидания: чистый лёд, дружелюбный гид и отличная организация.",
                   "Best honeymoon ever, the island resort was spotless and the staff was lovely."],
        "question": ["Нужна ли виза в Грузию для россиян и сколько можно находиться без неё? Кто ездил недавно?",
                     "Which month is best to visit Japan to see the cherry blossoms without huge crowds?"],
        "ad": ["Горящие туры в Турцию от 30 тысяч на двоих, всё включено, вылет в ближайшие дни.",
               "Book a week in the Alps now and get free ski passes for the whole family."],
        "news": ["С начала следующего месяца начнут действовать новые правила въезда для туристов, уведомили в посольстве.",
                 "The national airline will launch direct flights to Lisbon starting in March."],
    },
    "cinema": {
        "complaint": ["Потратил вечер на новый блокбастер и очень пожалел: сюжет дырявый, диалоги картонные, продолжение никому не нужно.",
                      "The cinema sold us tickets for a sold-out show and then the sound was broken for half the film."],
        "praise": ["Посмотрел наконец «Дюну» на большом экране: потрясающая картинка и музыка, рекомендую всем.",
                   "This indie drama is a masterpiece, the acting alone is worth the ticket."],
        "question": ["Что посмотреть вечером после хорошего триллера в духе «Семь»? Хочется чего-то мрачного и умного.",
                     "Does anyone know if the director's cut of this film is better than the theatrical one?"],
        "ad": ["Абонемент в кинотеатр на год со скидкой: два билета каждый месяц и попкорн в подарок.",
               "Stream thousands of movies in 4K, first month free for new subscribers."],
        "news": ["Каннский фестиваль объявил программу: в конкурсе примут участие двадцать картин из пятнадцати стран.",
                 "The studio confirmed that the sequel will be released in December with the original cast."],
    },
    "music": {
        "complaint": ["Билеты на концерт стоили как половина зарплаты, а звук в зале был ужасный, ничего не разобрать.",
                      "The streaming service keeps removing songs from my playlists without asking."],
        "praise": ["Новый альбом просто огонь: каждый трек цепляет, продюсирование на высоте. Слушаю по кругу.",
                   "Saw the quartet live last night, the whole hall was in tears. Wonderful."],
        "question": ["С чего начать изучение игры на гитаре взрослому человеку, какие аккорды учить первыми?",
                     "What is a good affordable MIDI keyboard for a beginner producer?"],
        "ad": ["Уроки вокала онлайн: поставим голос за три месяца, пробное занятие бесплатно.",
               "Vinyl record sale this weekend, up to 50% off on classic rock albums."],
        "news": ["Известный композитор получил премию за лучшую оригинальную музыку к фильму, церемония прошла в Лондоне.",
                 "The festival lineup was announced today, with sixty artists across four stages."],
    },
    "videogames": {
        "complaint": ["Игру выпустили в таком состоянии, что на старте вылетает каждые десять минут, а разработчики только обещают патч.",
                      "Paid full price for a game that is full of microtransactions and server crashes."],
        "praise": ["Прошёл ролевую игру года за сто часов и не заметил, как пролетело время: сюжет, мир и персонажи великолепны.",
                   "Such a charming little platformer, the controls feel perfect."],
        "question": ["Какой процессор нужен, чтобы новая стратегия шла на ультрах и не тормозила в больших битвах?",
                     "Anyone know how to unlock the secret ending in this game?"],
        "ad": ["Распродажа в магазине игр: до 80% скидки на хиты прошлого года, предложение действует до воскресенья.",
               "Pre-order the deluxe edition now and get exclusive in-game skins."],
        "news": ["Студия анонсировала дату выхода долгожданного продолжения и показала новый трейлер на выставке.",
                 "The publisher said the online mode will be shut down in June after ten years."],
    },
    "books": {
        "complaint": ["Купил новую книгу автора, а там сплошная вода и опечатки в каждом абзаце. Издательство, вам не стыдно?",
                      "Disappointed: the paperback arrived with pages missing and a torn cover."],
        "praise": ["Перечитала «Мастера и Маргариту» и снова влюбилась, каждая страница как маленькое чудо.",
                   "Highly recommend this novel, I could not put it down for two days."],
        "question": ["Посоветуйте, что почитать после «Охотника за разумом» в похожем жанре, но менее мрачное?",
                     "Is it worth reading the whole fantasy series or stop after the first three books?"],
        "ad": ["Книжный клуб приглашает: скидка 25% на первую покупку и бесплатная доставка при заказе от трёх книг.",
               "New cookbook out now, order today and get a signed bookplate."],
        "news": ["Литературная премия присуждена молодому прозаику за дебютный роман, сообщили организаторы.",
                 "The author announced that the final volume of the trilogy will be published in spring."],
    },
    "education": {
        "complaint": ["В университете опять перенесли сессию в последний момент, а деканат не отвечает на письма уже месяц.",
                      "The professor never replies to emails and grades are posted weeks late."],
        "praise": ["Очень рад, что поступил на этот факультет: сильные преподаватели, живые семинары и реальные проекты.",
                   "The online course was clear, well paced and genuinely changed my career."],
        "question": ["Подскажите, как лучше готовиться к вступительным по математике за полгода с нуля?",
                     "Is it better to do a master's right after the bachelor's or work for a year first?"],
        "ad": ["Подготовительные курсы к поступлению: небольшие группы, опытные преподаватели, первое занятие бесплатно.",
               "Enrol in our evening MBA programme and get 10% off for early applicants."],
        "news": ["Министерство объявило о новых правилах приёма в вузы: увеличено количество бюджетных мест на технические направления.",
                 "The university opened a new research centre with funding from the state."],
    },
    "pets": {
        "complaint": ["Ветеринарная клиника взяла деньги за полный анализ, а результаты потеряла, и собаку пришлось колоть заново.",
                      "My neighbour's dog barks all night and the owner just shrugs."],
        "praise": ["Спасибо приюту: взяли котёнка, он здоровый, привитый и уже освоился. Прекрасные люди работают.",
                   "Our rescue dog is the best thing that ever happened to us."],
        "question": ["Чем кормить британского котёнка в четыре месяца — сухим кормом или натуралкой?",
                     "How often should a puppy be taken for walks and when does vaccination start?"],
        "ad": ["Отдам котят в добрые руки, привиты, приучены к лотку, мальчики и девочки. Звоните.",
               "Premium dog food now 20% off, free shipping on your first order."],
        "news": ["В городе открылся новый приют для бездомных животных, волонтёры приглашают на день открытых дверей.",
                 "Authorities introduced mandatory microchipping for dogs starting next year."],
    },
    "cars": {
        "complaint": ["Купил новый автомобиль, а через неделю загорелась лампа двигателя; дилер разводит руками и отправляет к другому дилеру.",
                      "The mechanic charged me for a repair he never did, and the engine still rattles."],
        "praise": ["Электромобиль приятно удивил: запас хода реальный, разгон отличный, а обслуживание почти ничего не стоит.",
                   "This hatchback is economical and surprisingly roomy, I love it."],
        "question": ["Какой кроссовер взять до двух миллионов рублей для семьи из четырёх человек, чтобы не ломался?",
                     "Should I change winter tires now or wait until the first snow?"],
        "ad": ["Продаю автомобиль, 2019 года, один владелец, полный комплект резины, торг у капота.",
               "Test drive our new hybrid sedan this weekend and win a free service."],
        "news": ["Производитель отозвал более ста тысяч машин из-за дефекта подушек безопасности, сообщили в ведомстве.",
                 "The government will subsidise electric cars starting next quarter."],
    },
    "fashion": {
        "complaint": ["Заказала платье по размеру из таблицы, пришло на два размера меньше, а магазин отказывается принимать возврат.",
                      "The sneakers fell apart after two weeks. The brand's quality has dropped badly."],
        "praise": ["Новая коллекция просто восторг: качественные ткани, хороший крой, носится с удовольствием.",
                   "Love this jacket, warm and stylish, and it fits perfectly."],
        "question": ["Какой цвет пальто сейчас в моде и с чем его лучше носить, чтобы не выглядеть старомодно?",
                     "Where can I find a good tailor to shorten a suit jacket?"],
        "ad": ["Скидки до 70% на зимнюю коллекцию верхней одежды, доставка бесплатная, примерка при получении.",
               "Our designer handbags are now available in the online store, free returns."],
        "news": ["На неделе моды в Милане показали осенние коллекции, главный тренд сезона — тёплые оттенки и свободные силуэты.",
                 "A major retailer announced it will stop selling fur products from next year."],
    },
    "fitness": {
        "complaint": ["Абонемент в фитнес-клуб продали на год, а через два месяца закрыли зал и не вернули ни копейки.",
                      "The gym is always overcrowded and half the machines are broken."],
        "praise": ["Три месяца занимаюсь с новым тренером: минус восемь килограммов и никаких травм, очень доволен.",
                   "This yoga studio is peaceful and the teachers are fantastic."],
        "question": ["Сколько раз в неделю нужно ходить в зал новичку, чтобы набрать мышечную массу и не перетренироваться?",
                     "What is the best way to build up to a first 10k run?"],
        "ad": ["Годовой абонемент в фитнес-клуб со скидкой 40%, бассейн и сауна в подарок, акция до пятницы.",
               "Join our bootcamp this month and get your first week free."],
        "news": ["В городе пройдёт массовый забег на десять километров, регистрация участников открыта до конца недели.",
                 "The national federation introduced new fitness standards for school children."],
    },
    "basketball": {
        "complaint": ["Судьи в этом матче просто убили игру: десяток спорных фолов и ни одного в пользу гостей. Позор лиги.",
                      "Tickets to the playoff game were overpriced and the arena had awful sightlines."],
        "praise": ["Какой матч! Разыгрывающий выдал трипл-дабл и вытащил команду в овертайме, настоящий лидер.",
                   "Brilliant defence from the whole team tonight, what a season so far."],
        "question": ["Как правильно поставить бросок со средней дистанции, чтобы не заваливать локоть?",
                     "Who do you think will win the MVP award this year?"],
        "ad": ["Набор в детскую баскетбольную секцию, тренеры-профессионалы, первое занятие бесплатно.",
               "Official team jerseys now on sale, get 15% off with the club code."],
        "news": ["Клуб подписал контракт с центровым из Европы на два сезона, сумма сделки не разглашается.",
                 "The league announced that the all-star game will be held in Chicago."],
    },
    "psychology": {
        "complaint": ["Сходила к психологу, а он всю сессию смотрел на часы и в конце сказал просто «терпите». Деньги на ветер.",
                      "The therapist cancelled three sessions in a row without notice."],
        "praise": ["Терапия реально помогла: стало легче справляться с тревогой, спасибо моему специалисту.",
                   "That book on habits changed how I think about motivation."],
        "question": ["Как понять, что это уже выгорание, а не просто усталость, и когда стоит обращаться к специалисту?",
                     "How can I stop overthinking every conversation after it happens?"],
        "ad": ["Онлайн-консультации психолога: подберём специалиста за один день, первая встреча со скидкой 50%.",
               "Mindfulness workshop this Saturday, limited seats, sign up today."],
        "news": ["Исследователи опубликовали результаты большого исследования о влиянии сна на эмоциональное состояние подростков.",
                 "A new study links daily screen time to anxiety in young adults."],
    },
    "space": {
        "complaint": ["Запуск переносят в который раз, а агентство даже не объясняет причины. Налогоплательщики вправе знать.",
                      "Half of the budget was wasted on a mission that never left the launch pad."],
        "praise": ["Потрясающие снимки с нового телескопа: далёкие галактики видны так чётко, будто рядом. Браво, учёные.",
                   "That landing was flawless, incredible engineering."],
        "question": ["Сколько времени нужно, чтобы долететь до Марса при нынешних технологиях и можно ли сократить путь?",
                     "Why do astronauts need to exercise for hours every day on the station?"],
        "ad": ["Экскурсия в планетарий с лекцией и ночными наблюдениями, билеты со скидкой для школьных групп.",
               "Buy a telescope for beginners today, free star map included."],
        "news": ["Космический аппарат успешно вышел на орбиту Луны и начал передавать первые данные на Землю.",
                 "Scientists discovered a new exoplanet in the habitable zone of a nearby star."],
    },
    "programming": {
        "complaint": ["После обновления библиотеки всё сломалось, документация устарела, а мейнтейнеры закрыли issue без комментариев.",
                      "The build takes forty minutes and the tests are flaky. Nobody on the team cares."],
        "praise": ["Перешли на новый фреймворк, и скорость разработки выросла вдвое: отличная документация и понятный API.",
                   "This editor plugin saves me an hour every day, brilliant work."],
        "question": ["Подскажите, как правильно обрабатывать исключения в асинхронном коде на Python, чтобы не терять трейсбек?",
                     "What is the difference between a mutex and a semaphore in practice?"],
        "ad": ["Курсы по программированию на Python с трудоустройством: старт группы в понедельник, рассрочка без процентов.",
               "Try our cloud IDE free for 30 days, fast builds and team collaboration."],
        "news": ["Вышла новая версия языка с улучшенной типизацией и ускоренным интерпретатором, сообщили разработчики.",
                 "The open-source project moved to a new licence after months of discussion."],
    },
    "security": {
        "complaint": ["Сервис допустил утечку базы паролей и две недели молчал, а теперь предлагает «сменить пароль» вместо извинений.",
                      "Support told me to disable two-factor authentication to fix a login bug. Terrible advice."],
        "praise": ["Менеджер паролей наконец-то удобный: автозаполнение работает везде, а все пароли зашифрованы локально.",
                   "Great security audit report, clear findings and practical fixes."],
        "question": ["Как проверить, не взломан ли мой аккаунт, и что делать, если пришло письмо о входе из другой страны?",
                     "Is a VPN enough to stay safe on public Wi-Fi?"],
        "ad": ["Антивирус на три устройства за 999 рублей в год, защита от шифровальщиков и фишинга.",
               "Penetration testing for small business, book a free consultation."],
        "news": ["Хакеры атаковали крупный банк, данные клиентов не пострадали, сообщили в пресс-службе.",
                 "Researchers found a critical vulnerability in a popular router firmware."],
    },
    "ai": {
        "complaint": ["Чат-бот поддержки не понимает ни одного вопроса и ходит по кругу. Это не помощь, а издевательство.",
                      "The AI image generator keeps ignoring my prompt and produces nonsense."],
        "praise": ["Нейросеть для перевода работает удивительно точно, даже сленг и шутки передаёт правильно.",
                   "Impressed by how well this assistant summarises long documents."],
        "question": ["Как объяснить школьнику, чем нейросеть отличается от обычной программы, простыми словами?",
                     "Can a language model run locally on a laptop without a GPU?"],
        "ad": ["Курс по машинному обучению с практикой на реальных данных, старт набора в следующем месяце.",
               "Our AI writing tool is free for the first 7 days. Try it now."],
        "news": ["Исследователи представили новую модель, которая лучше справляется с логическими задачами, чем предыдущая версия.",
                 "The company announced a partnership to deploy AI assistants in hospitals."],
    },
    "restaurants": {
        "complaint": ["В ресторане ждали заказ час сорок, принесли холодное, а в счёт ещё добавили сервис, о котором никто не предупреждал.",
                      "Found a hair in my soup and the manager just rolled his eyes."],
        "praise": ["Отличное место: тёплая атмосфера, вежливый персонал и пасту делают лучше, чем в Италии.",
                   "Lovely little cafe, the coffee and the cheesecake were perfect."],
        "question": ["Подскажите, где в центре можно вкусно поужинать вдвоём, чтобы было тихо и без громкой музыки?",
                     "Is it necessary to reserve a table at this place on weekends?"],
        "ad": ["Бизнес-ланч в кафе у метро: суп, горячее и компот за 350 рублей, доставка в офисы бесплатно.",
               "Celebrate your birthday with us: free dessert and a 20% discount for the group."],
        "news": ["В городе открылся новый ресторан от известного шефа, меню строится вокруг местных продуктов.",
                 "The popular chain announced it will open ten new cafes across the region."],
    },
    "telecom": {
        "complaint": ["Провайдер обещал гигабит, а скорость едва добирается до двадцати мегабит; поддержка только присылает шаблонные ответы.",
                      "My mobile operator keeps charging for services I never ordered."],
        "praise": ["Очень доволен новым тарифом: безлимитный интернет работает без ограничений, связь стабильная даже за городом.",
                   "Fibre installation took an hour and the connection has been rock solid."],
        "question": ["Как перенести номер к другому оператору и сохранить баланс на счёте? Кто делал недавно?",
                     "What router should I buy for a three-room flat with thick walls?"],
        "ad": ["Подключите домашний интернет и телевидение за 500 рублей в месяц, скидка для новых абонентов.",
               "Switch to our network and get 50 GB for free during the first three months."],
        "news": ["Оператор запустил сеть пятого поколения ещё в десяти городах, покрытие охватит центральные районы.",
                 "Regulators approved the merger of two mobile operators with conditions."],
    },
    "taxes": {
        "complaint": ["Налоговая прислала требование по уже оплаченному налогу и три недели не может найти платёж. Бюрократия.",
                      "The tax office keeps asking for the same documents and never confirms receipt."],
        "praise": ["Юрист по налогам помог вернуть вычет за лечение за две недели, всё чётко и без лишних вопросов.",
                   "The new tax portal is much easier to use than the old one."],
        "question": ["Нужно ли платить налог с продажи квартиры, если владел ей меньше пяти лет, и как оформить вычет?",
                     "How do I register as self-employed and what taxes do I pay?"],
        "ad": ["Бухгалтерское сопровождение ИП и ООО от 3000 рублей в месяц, сдача отчётности и консультации юриста.",
               "File your taxes online with our software and get a guaranteed maximum refund."],
        "news": ["Правительство внесло в парламент законопроект об изменении ставок налога для малого бизнеса.",
                 "The finance ministry announced an extension of the deadline for filing returns."],
    },
    "parenting": {
        "complaint": ["В детском саду воспитатели опять не следят за группой, а заведующая отвечает, что «все дети падают». Как это терпеть?",
                      "The school keeps sending homework late at night and expects parents to do it."],
        "praise": ["Очень благодарны учительнице: она нашла подход к нашему застенчивому сыну, и он наконец полюбил школу.",
                   "This parenting podcast has made bedtime so much easier."],
        "question": ["Как отучить трёхлетнего ребёнка от соски без слёз и истерик, кто через это проходил?",
                     "At what age should children start having their own phone?"],
        "ad": ["Развивающие занятия для детей от года до шести лет, малые группы, пробный урок бесплатно.",
               "Order a baby stroller now and receive a free changing bag."],
        "news": ["В регионе увеличат пособия многодетным семьям и откроют ещё сто мест в яслях.",
                 "A new law gives working parents extra days of paid leave per year."],
    },
    "gardening": {
        "complaint": ["Купил саженцы в питомнике, а они все засохли за неделю; продавец говорит, что я неправильно поливал.",
                      "The neighbour's tree overhangs my garden and he refuses to trim it."],
        "praise": ["Новая теплица радует: помидоры созрели на месяц раньше, и никакой фитофторы. Рекомендую.",
                   "This compost mix did wonders for the roses."],
        "question": ["Когда лучше сажать тюльпаны осенью и на какую глубину закапывать луковицы?",
                     "How do I get rid of aphids on pepper plants without chemicals?"],
        "ad": ["Продаю саженцы плодовых деревьев и кустарников, доставка по области, скидки при оптовых заказах.",
               "Garden tools sale this weekend, everything half price."],
        "news": ["В городском парке открыли новый ботанический сад, вход свободный по выходным.",
                 "Farmers reported an early harvest this season thanks to the warm spring."],
    },
    "renovation": {
        "complaint": ["Бригада сорвала сроки ремонта на два месяца, а теперь требует доплату за материалы, которые уже были в смете.",
                      "The contractor left the bathroom unfinished and stopped answering the phone."],
        "praise": ["Мастера сделали ремонт аккуратно и в срок, оставили после себя чистоту. Рекомендую от души.",
                   "The new kitchen turned out beautifully, the fitters were very professional."],
        "question": ["Какую стяжку лучше делать под тёплый пол в панельном доме и сколько она сохнет?",
                     "Do I need a permit to knock down a non-load-bearing wall?"],
        "ad": ["Ремонт квартир под ключ, гарантия 3 года, фиксированная смета, выезд замерщика бесплатно.",
               "Premium laminate flooring on sale, free delivery to your door."],
        "news": ["В городе вступили в силу новые правила согласования перепланировок квартир, сообщили в администрации.",
                 "The government extended the programme for subsidising building renovations."],
    },
    "utilities": {
        "complaint": ["Управляющая компания в третий раз выставила счёт за отопление в самое тёплое время года, а перерасчёт не делает.",
                      "No hot water for two weeks and the bill is still full price."],
        "praise": ["Спасибо управляющей компании: подъезд отремонтировали, двор убрали, а на обращения отвечают в тот же день.",
                   "The new smart meters are accurate and easy to read."],
        "question": ["Как передать показания счётчиков воды без визита в офис и в какие числа это нужно делать?",
                     "Why did our electricity bill double this month?"],
        "ad": ["Поверка счётчиков воды на дому за 500 рублей, выезд в день обращения, оформление документов.",
               "Switch to our green energy plan and cut your bill by 15%."],
        "news": ["С первого числа в регионе вырастут тарифы на коммунальные услуги в среднем на семь процентов.",
                 "The city announced planned water outages for maintenance this weekend."],
    },
    "transport": {
        "complaint": ["Автобус не приходит по полчаса, а в часы пик набит так, что невозможно зайти. Городские власти, где реформа?",
                      "The metro line has been delayed every single morning this week."],
        "praise": ["Новая трамвайная линия — это чудо: быстро, чисто, по расписанию, и до работы теперь десять минут.",
                   "The new night buses are a real improvement."],
        "question": ["Подскажите, как оформить единый проездной на метро и автобусы и сколько он стоит в месяц?",
                     "Is there a direct train from the airport to the city centre?"],
        "ad": ["Продам электросамокат для поездок по городу, пробег небольшой, зарядное и замок в комплекте.",
               "Monthly transit passes for students available at a discount."],
        "news": ["С понедельника на двух линиях метро изменится расписание: интервалы между поездами сократят до трёх минут.",
                 "The city will introduce free bus rides for pensioners on weekends."],
    },
    "shopping": {
        "complaint": ["Заказ в интернет-магазине оплатил, а товара нет, на сайте «в обработке» уже месяц, деньги не возвращают.",
                      "The seller sent a counterfeit and then blocked me."],
        "praise": ["Удобный магазин: быстрый поиск, честные отзывы и возврат без вопросов. Заказываю уже третий раз.",
                   "Great online store, the item matched the photos exactly."],
        "question": ["Как отличить оригинальные наушники от подделки при покупке на маркетплейсе?",
                     "Is it safe to pay by card on a website I have never used before?"],
        "ad": ["Распродажа в интернет-магазине: скидки до 60% на бытовую технику, бесплатная доставка от 3000 рублей.",
               "Use promo code SAVE10 for 10% off your first order."],
        "news": ["Крупный маркетплейс объявил о запуске пунктов выдачи в малых городах и новых правилах возврата.",
                 "Online retail sales grew by a quarter compared to last year, analysts said."],
    },
    "support": {
        "complaint": ["Служба поддержки третий день отвечает одними шаблонами и перекидывает меня от оператора к оператору. Никакого решения.",
                      "Waited two hours on hold and then the call dropped. Awful customer service."],
        "praise": ["Спасибо поддержке: оператор разобралась в моей проблеме за пять минут и ещё предложила компенсацию.",
                   "The support agent was friendly and fixed the issue on the first call."],
        "question": ["Как связаться с живым оператором, если чат-бот не понимает мой вопрос и просит выбрать пункт меню?",
                     "Does anyone know the working hours of this company's helpline?"],
        "ad": ["Аутсорсинг службы поддержки для интернет-магазинов: ответим клиентам за 5 минут, первые две недели бесплатно.",
               "Our help desk software handles tickets for small teams, try it free."],
        "news": ["Компания запустила круглосуточную линию поддержки и начала принимать обращения через мессенджеры.",
                 "The telecom company said it would add 500 operators to cut waiting times."],
    },
    "photography": {
        "complaint": ["Фотограф на свадьбе опоздал на час, а потом прислал снимки в ужасном качестве и ещё просит доплатить.",
                      "The camera shop sold me a lens with a scratch and refuses to exchange it."],
        "praise": ["Фотосессия получилась прекрасная: фотограф расслабил нас, и кадры выглядят живыми и тёплыми.",
                   "This mirrorless camera takes stunning low-light shots."],
        "question": ["Какой объектив взять для портретов на кроп-камеру, чтобы размытый фон был красивым и недорого?",
                     "How do I stop my photos from looking flat in harsh midday sun?"],
        "ad": ["Семейные фотосессии на природе и в студии, скидка 20% на съёмку в будни, готовые фото через три дня.",
               "Photo editing presets pack, 30 styles for portraits and landscapes, on sale today."],
        "news": ["В музее открылась выставка работ известного фотографа, экспозиция продлится до конца осени.",
                 "The photography contest announced the winners from over ten thousand entries."],
    },
}

TEXTS_HELD: Dict[str, Dict[str, List[str]]] = {
    "delivery": {
        "complaint": ["Курьер принёс заказ спустя четыре часа, коробка была вскрыта, а половины товаров не хватало. Службе доставки стыдно должно быть.",
                      "My parcel has been stuck at the sorting centre for ten days and nobody tells me anything."],
        "praise": ["Привезли за полчаса, горячее и упакованное идеально, курьер вежливый. Отличная доставка, буду заказывать ещё.",
                   "Super fast shipping, the package arrived a day early and well protected."],
        "question": ["Подскажите, как отследить посылку из-за границы, если трек-номер не находится на сайте почты?",
                     "Can I change the delivery address after placing an order?"],
        "ad": ["Бесплатная доставка по городу при заказе от 1000 рублей, доставим за два часа или вернём деньги.",
               "Same-day delivery for all orders placed before noon, try it today."],
        "news": ["Почта объявила, что сроки доставки писем в сельские районы сократят до трёх дней с начала следующего года.",
                 "The courier company expanded its service to twenty new cities this month."],
    },
    "recipes": {
        "complaint": ["Попробовала рецепт из этого блога — тесто не поднялось, пропорции неверные, потратила продукты зря. Автор, проверяйте рецепты.",
                      "The recipe said bake for twenty minutes but the cake was raw inside after an hour."],
        "praise": ["Спасибо за рецепт запечённой курицы: получилось сочно, хрустящая корочка, вся семья в восторге.",
                   "This soup recipe is perfect, simple and so flavourful."],
        "question": ["Чем можно заменить сливки в рецепте карбонары, если хочется сделать полегче?",
                     "How long should I rest dough before rolling it out for pasta?"],
        "ad": ["Мастер-класс по итальянской кухне: научим готовить пасту и пиццу, запись открыта, скидка по промокоду.",
               "Buy our premium cookware set and make restaurant-quality dishes at home."],
        "news": ["В столице пройдёт гастрономический фестиваль, на котором шеф-повара представят авторские блюда из местных продуктов.",
                 "A famous chef announced a new cooking show launching next month."],
    },
    "football": {
        "complaint": ["Тренер снова выставил слабый состав, а судья не заметил явный пенальти. Болельщики имеют право на нормальный футбол.",
                      "Paid a fortune for the match ticket and the stadium toilets were closed. Disgraceful organisation."],
        "praise": ["Какой гол! Дальний удар под перекладину на девяносто второй минуте, весь стадион встал. Браво команде.",
                   "What a performance from the young midfielder, he ran the game."],
        "question": ["Кто, по-вашему, выиграет чемпионат в этом сезоне, и стоит ли ставить на аутсайдера?",
                     "Can someone explain the offside rule in simple words?"],
        "ad": ["Абонементы на домашние матчи со скидкой 25% до конца месяца, детям — бесплатно по выходным.",
               "Official club shirts available now, order online with free delivery."],
        "news": ["Футбольный клуб объявил о назначении нового главного тренера, контракт подписан на три года.",
                 "The federation confirmed the date of the cup final and the host city."],
    },
    "gadgets": {
        "complaint": ["Новый смартфон начал греться и вылетать через месяц, сервисный центр говорит, что гарантия не распространяется. Обман.",
                      "The earbuds stopped charging after three weeks and support refuses to replace them."],
        "praise": ["Взял этот ноутбук для учёбы и не нарадуюсь: лёгкий, тихий, экран яркий, батарея держит целый день.",
                   "Best smartwatch I have owned, accurate sensors and a battery that lasts a week."],
        "question": ["Что лучше взять для фото и видео: флагман прошлого года или новую модель среднего класса?",
                     "How do I transfer all my data from an old phone to a new one?"],
        "ad": ["Скидка на ноутбуки и планшеты до 25%, рассрочка без переплат, бесплатная доставка по городу.",
               "Pre-order the new smart speaker and receive a free wireless charger."],
        "news": ["Компания представила новый флагманский смартфон с улучшенной камерой, продажи начнутся в ноябре.",
                 "The manufacturer announced a firmware update that extends the battery life of its headphones."],
    },
    "politics": {
        "complaint": ["Власти опять приняли решение без обсуждения с жителями, а на публичных слушаниях нас даже не стали слушать. Возмутительно.",
                      "The council ignored thousands of petitions and passed the plan in a closed meeting."],
        "praise": ["Хорошее решение мэрии: наконец отремонтируют дороги и откроют консультации для жителей. Поддерживаю.",
                   "The new minister handled the crisis well and communicated clearly."],
        "question": ["Как зарегистрироваться на дистанционное электронное голосование и нужна ли для этого электронная подпись?",
                     "What is the difference between a presidential and a parliamentary system?"],
        "ad": ["Партия приглашает на встречу с кандидатом: расскажем о программе и ответим на вопросы жителей района.",
               "Subscribe to our political newsletter for daily analysis, first month free."],
        "news": ["Парламент одобрил в первом чтении законопроект о налоговых льготах для малого бизнеса, сообщил председатель комитета.",
                 "The president met with the heads of regional governments to discuss the budget."],
    },
    "jobs": {
        "complaint": ["Работодатель задерживает зарплату уже третий месяц и угрожает увольнением тем, кто напоминает о долге. Это просто беспредел.",
                      "The recruiter ghosted me after four interview rounds without any feedback."],
        "praise": ["Очень доволен новой работой: адекватное руководство, прозрачные премии, гибкий график и внимательные коллеги.",
                   "My team is wonderful, I look forward to Mondays for the first time."],
        "question": ["Как лучше составить резюме без опыта работы, на что обращать внимание работодателя?",
                     "Should I accept a counter-offer from my current employer?"],
        "ad": ["Требуются курьеры и сборщики заказов, график свободный, выплаты каждую неделю, работа рядом с домом.",
               "We are hiring a front-end developer, remote, salary above market."],
        "news": ["Уровень безработицы в стране снизился до исторического минимума, сообщила статистическая служба.",
                 "The company announced that it would open two thousand new jobs next year."],
    },
}


# ---------------------------------------------------------------------------------------------------------------------------------
def _is_cyrillic(text: str) -> bool:
    return any("а" <= ch.lower() <= "я" or ch in "ёЁ" for ch in text)


def _fmt(tpl: str, names: Dict[str, str]) -> str:
    return tpl.format(nom=names["nom"], acc=names["acc"], en=names["en"])


def criteria(topic: str, tone: str, topics: Dict[str, Dict[str, str]]) -> List[Tuple[str, str, bool, str, Optional[str], Optional[str]]]:
    """Every (kind, question, truth, lang, question_topic, question_tone) a text of (topic, tone) can be asked about; the question's own
    topic / tone are None for the parts a kind does not name."""
    out: List[Tuple[str, str, bool, str, Optional[str], Optional[str]]] = []
    for lang in ("ru", "en"):
        for t, names in topics.items():
            for q in TOPIC_QUESTIONS[lang]:
                out.append(("topic", _fmt(q, names), t == topic, lang, t, None))
        for u in TONES:
            for q in TONE_QUESTIONS[u][lang]:
                out.append(("tone", q, u == tone, lang, None, u))
            for t, names in topics.items():
                for q in COMPOUND_QUESTIONS[u][lang]:
                    out.append(("compound", _fmt(q, names), (u == tone and t == topic), lang, t, u))
    return out


def build(split: str = "train", seed: int = 0, topics: Optional[Dict[str, Dict[str, str]]] = None,
          texts: Optional[Dict[str, Dict[str, List[str]]]] = None) -> List[Sample]:
    """Yes/no Samples over the text bank.  Per text: a few matching questions and about twice as many non-matching ones, drawn so that they
    over-represents the hard negatives of the compound kind -- the same topic with another tone and the same tone with another topic --
    so a compound question cannot be answered from its topic or its tone alone.  The question language follows the text language,
    with a quarter of the other language mixed in."""
    topics = topics or (TOPICS_TRAIN if split == "train" else TOPICS_HELD)
    texts = texts or (TEXTS_TRAIN if split == "train" else TEXTS_HELD)
    rng = random.Random(seed)
    samples: List[Sample] = []
    for topic, by_tone in texts.items():
        for tone, items in by_tone.items():
            for text in items:
                lang = "ru" if _is_cyrillic(text) else "en"
                pool = [c for c in criteria(topic, tone, topics) if c[3] == lang or rng.random() < 0.25]
                # positives: one paraphrase per kind (a second with probability 1/2), so that the question wording varies and the
                # share of "yes" stays about a third -- a feed has more items that do not match than items that do
                chosen = []
                for kind in ("topic", "tone", "compound"):
                    of_kind = [c for c in pool if c[2] and c[0] == kind]
                    if of_kind:
                        chosen += rng.sample(of_kind, min(len(of_kind), 2 if rng.random() < 0.5 else 1))
                neg = [c for c in pool if not c[2]]
                same_topic = [c for c in neg if c[0] == "compound" and c[4] == topic]
                same_tone = [c for c in neg if c[0] == "compound" and c[5] == tone]
                easy_topic = [c for c in neg if c[0] == "topic"]
                easy_tone = [c for c in neg if c[0] == "tone"]
                k = max(2, 2 * len(chosen))
                for bucket, share in ((same_topic, 0.30), (same_tone, 0.25), (easy_topic, 0.20), (easy_tone, 0.25)):
                    if bucket:
                        chosen += rng.sample(bucket, min(len(bucket), max(1, int(round(k * share)))))
                for kind, q, truth, qlang, _qt, _qu in chosen:
                    samples.append(Sample(task="content_" + kind, family="content", lang=lang if qlang == lang else "mix",
                                          ptype="noul", state=text, question=q, target=1.0 if truth else 0.0, split=split,
                                          meta={"topic": topic, "tone": tone, "kind": kind, "synth": True, "lic": "own"}))
    rng.shuffle(samples)
    return samples
