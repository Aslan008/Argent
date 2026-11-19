"""What the teacher may write about, and what is reserved for the exams (hold-out by FAMILY, plan section 5.3).

Each pool entry is (family code, description). The queue builder samples batches only from TRAIN pools; the HOLDOUT families
are never shown to the generator, and the ingest refuses any record that claims one of their codes. Leeth 3 exams are written
by a different model family on exactly these held-out families, so an exam measures transfer, not recall.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Pool = List[Tuple[str, str]]

SAFETY_TRAIN: Pool = [
    ("coreutils", "Linux/macOS: файлы и каталоги (rm, mv, cp, find, ln, truncate, shred, tar)"),
    ("pkg-managers", "пакетные менеджеры: apt, dnf, pacman, brew, npm, pnpm, yarn, pip, uv, cargo, gem, composer, nuget, winget, choco, conda"),
    ("git", "git: commit, reset, rebase, push, clean, stash, branch -D, tag, remote, gc"),
    ("containers", "docker, podman, kubectl, helm, docker compose"),
    ("databases", "psql, mysql, sqlite3, redis-cli, mongosh, SQL-запросы и миграции"),
    ("cloud-cli", "aws, gcloud, az, terraform, pulumi, s3/blob-операции"),
    ("powershell", "PowerShell: Remove-Item, Set-*, Get-*, Stop-Process, Invoke-WebRequest, реестр"),
    ("windows-cmd", "cmd: del, rd, format, reg, schtasks, netsh, attrib, robocopy, bcdedit, shutdown"),
    ("macos-admin", "macOS: launchctl, defaults, diskutil, tmutil, csrutil, spctl"),
    ("network", "curl, wget, ssh, scp, rsync, iptables, ufw, nmap, ping, nslookup"),
    ("system", "kill, pkill, systemctl, shutdown, chmod, chown, sudo, mount, dd, mkfs, swapoff, useradd"),
    ("browser-actions", "действия агента в браузере: клики по кнопкам оплаты/удаления/отправки, cookies, формы, загрузки"),
    ("file-manager", "действия с файлами через API агента: удалить, переместить, перезаписать, очистить корзину"),
    ("deploy", "публикация и деплой: npm publish, twine upload, gh release, vercel/netlify deploy, docker push"),
    ("messaging-api", "отправка писем/сообщений/уведомлений через API, рассылки"),
    ("payments", "действия в магазине/банке/подписках: оплатить, подписаться, отменить, вернуть"),
    ("scripts-exec", "выполнение скриптов: python/node/bash-скрипты проекта и скачанные, curl | sh, iex"),
    # Admin areas where the exam measures transfer to UNKNOWN programs: the area is trained with other programs (the exam's programs
    # are refused by SAFETY_HOLDOUT_PROGRAMS), so the model learns what an action does, not which program names are dangerous.
    ("storage-admin", "диски, RAID и тома: mdadm, lsblk, blkid, lvs/vgs/pvs, lvextend, resize2fs, e2fsck, xfs_repair, btrfs, Storage Spaces (PowerShell)"),
    ("backup-tools", "резервные копии и снимки: duplicity, rsnapshot, kopia, bconsole, wbadmin, Veeam CLI, снимки дисков в облаке"),
    ("keys-secrets", "ключи и секреты: ssh-keygen, ssh-add, age, keytool, step, vault, pass, секреты Kubernetes и облаков"),
    ("audit-logs", "журналы безопасности и аудита: auditpol, Get-WinEvent, Clear-EventLog, ausearch, aureport, aa-enforce, fail2ban-client"),
    ("boot-kernel", "загрузка и ядро: grub-mkconfig, update-grub, dracut, mkinitcpio, kexec, ulimit, /proc/sys, cpupower, tuned-adm"),
    ("firmware-flash", "прошивка плат и устройств: fwupdmgr, avrdude, dfu-util, openocd, flashrom, arduino-cli, platformio"),
    ("fleet-mgmt", "управление парком машин: puppet, chef-client, pdsh, pssh, fabric, массовые команды по списку хостов"),
    ("schedulers", "планировщики заданий: at, systemd-run, таймеры systemd, anacron, launchd, Task Scheduler (PowerShell)"),
    ("acl-permissions", "права и владельцы: setfacl, getfacl, Set-Acl, Get-Acl, chown -R, umask, sudoers через visudo"),
]
SAFETY_HOLDOUT: Pool = [
    ("ide-editor", "команды редакторов и IDE: code, vim, emacs, JetBrains CLI"),
    ("iot-home", "умный дом и IoT: mosquitto_pub, Home Assistant API, прошивка устройств"),
    ("game-admin", "консоли администрирования игровых серверов (rcon, minecraft-server)"),
    ("legacy-held", "diskpart, wevtutil, takeown, icacls, crontab, cryptsetup, git filter-branch, sc stop, net stop"),
]

# Programs whose commands appear ONLY in the safety exams ("unknown programs"): the ingest refuses a safety record that names one of
# them as a word, whatever family the teacher wrote, so the exam measures transfer to programs the model never saw.
SAFETY_HOLDOUT_PROGRAMS = (
    "diskpart", "wevtutil", "takeown", "icacls", "crontab", "cryptsetup", "filter-branch", "vim", "emacs", "nvim",
    "mosquitto_pub", "mosquitto_sub", "hass-cli", "esptool", "esphome", "rcon", "rcon-cli", "steamcmd", "zfs", "zpool", "lvremove",
    "vgremove", "pvremove", "lvcreate", "parted", "fsutil", "vssadmin", "cipher", "gpg", "openssl", "rclone", "restic", "borg",
    "certbot", "ansible", "ansible-playbook", "salt", "wipefs", "sgdisk", "tune2fs", "efibootmgr", "auditctl", "setenforce", "sysctl",
    "modprobe", "nft", "firewall-cmd", "chattr", "logrotate", "journalctl", "badblocks", "smartctl", "hdparm",
)

TOOLS_TRAIN: Pool = [(c, d) for c, d in [
    ("email-client", "почтовый клиент"), ("calendar", "календарь и встречи"), ("crm", "CRM для продаж"),
    ("accounting-sme", "бухгалтерия малого бизнеса"), ("photo-editor", "фоторедактор"), ("video-editor", "видеоредактор"),
    ("podcast", "запись и монтаж подкастов"), ("ide", "среда разработки"), ("ci-cd", "CI/CD-панель"),
    ("cloud-console", "облачная консоль"), ("kanban", "канбан-доска задач"), ("wiki", "корпоративная вики"),
    ("chat-ops", "чат-бот для команды"), ("e-shop-admin", "админка интернет-магазина"), ("warehouse", "складской учёт"),
    ("delivery", "служба доставки"), ("taxi", "заказ такси"), ("food-order", "заказ еды"), ("banking", "мобильный банк"),
    ("brokerage-view", "просмотр портфеля (без сделок)"), ("hotel-booking", "бронирование отелей"), ("airline", "авиакомпания"),
    ("gym-booking", "запись в спортзал"), ("clinic", "запись в клинику"), ("school-lms", "школьная LMS"),
    ("hr-portal", "HR-портал"), ("recruiting", "система найма"), ("helpdesk", "служба поддержки"),
    ("smart-tv", "умный телевизор"), ("music-player", "музыкальный плеер"), ("maps", "карты и навигация"),
    ("note-taking", "заметки"), ("spreadsheet", "электронные таблицы"), ("presentation", "презентации"),
    ("pdf-tools", "работа с PDF"), ("password-manager", "менеджер паролей (без показа паролей)"),
    ("router-admin", "админка роутера"), ("nas", "сетевое хранилище"), ("printer", "принтер и сканер"),
    ("survey", "опросы и формы"), ("newsletter", "рассылки"), ("social-media", "соцсеть: публикации"),
    ("forum-mod", "модерация форума"), ("game-launcher", "игровой лаунчер"), ("car-app", "приложение автомобиля"),
    ("drone", "управление дроном (симулятор)"), ("3d-printer", "3D-принтер"), ("lab-inventory", "лабораторный учёт"),
    ("legal-docs", "юридические документы"), ("real-estate", "недвижимость: объявления"), ("insurance", "страховка"),
    ("tax-filing", "налоговая декларация"), ("library", "библиотека"), ("museum-guide", "аудиогид"),
    ("translation", "переводчик документов"), ("dictation", "диктовка и расшифровка"), ("security-cams", "камеры наблюдения"),
    ("parking", "парковка"), ("event-tickets", "билеты на мероприятия"), ("volunteer", "волонтёрская платформа"),
    ("farm", "агро-учёт"), ("restaurant-pos", "касса ресторана"), ("beauty-salon", "салон красоты"),
    ("pet-clinic", "ветклиника"), ("home-budget", "домашний бюджет"), ("travel-planner", "планировщик поездок"),
    ("devops-monitoring", "мониторинг серверов"), ("database-admin", "администрирование БД"), ("api-gateway", "API-шлюз"),
    ("analytics", "веб-аналитика"), ("ads-manager", "рекламный кабинет"), ("seo-tool", "SEO-инструмент"),
    ("code-review", "код-ревью"), ("package-registry", "реестр пакетов"), ("ml-platform", "платформа ML-экспериментов"),
]]
TOOLS_HOLDOUT: Pool = [
    ("astronomy", "обсерватория и телескоп"), ("shipping-port", "морской порт"), ("theatre", "театр: репертуар и билеты"),
    ("orchestra", "оркестр: ноты и репетиции"), ("botanical-garden", "ботанический сад"), ("bakery", "пекарня"),
    ("railway", "железная дорога"), ("ski-resort", "горнолыжный курорт"), ("archive", "госархив"),
    ("argent-agent", "агент-программист на ПК (инструменты Argent)"),
]

CONTENT_TRAIN: Pool = [(c, d) for c, d in [
    ("coffee", "кофе"), ("football", "футбол"), ("tennis", "теннис"), ("cycling", "велоспорт"), ("running", "бег"),
    ("cooking", "кулинария"), ("baking", "выпечка"), ("vegan", "веганство"), ("wine", "вино"), ("tea", "чай"),
    ("travel-europe", "путешествия по Европе"), ("travel-asia", "путешествия по Азии"), ("hiking", "походы"),
    ("camping", "кемпинг"), ("cars-ev", "электромобили"), ("cars-repair", "ремонт авто"), ("motorcycles", "мотоциклы"),
    ("smartphones", "смартфоны"), ("laptops", "ноутбуки"), ("pc-gaming", "игры на ПК"), ("board-games", "настольные игры"),
    ("programming-python", "программирование на Python"), ("web-dev", "веб-разработка"), ("ai-news", "новости ИИ"),
    ("crypto", "криптовалюты"), ("personal-finance", "личные финансы"), ("real-estate", "недвижимость"),
    ("parenting", "воспитание детей"), ("pregnancy", "беременность"), ("pets-dogs", "собаки"), ("pets-cats", "кошки"),
    ("gardening", "садоводство"), ("houseplants", "комнатные растения"), ("diy-home", "ремонт квартиры"),
    ("interior", "интерьер"), ("fashion", "мода"), ("skincare", "уход за кожей"), ("fitness", "фитнес"), ("yoga", "йога"),
    ("mental-health", "психологическое здоровье"), ("sleep", "сон"), ("nutrition", "питание"), ("medicine-general", "общая медицина"),
    ("dentistry", "стоматология"), ("education-school", "школьное образование"), ("university", "университет"),
    ("language-learning", "изучение языков"), ("books-fiction", "художественная литература"), ("movies", "кино"),
    ("series", "сериалы"), ("music-rock", "рок-музыка"), ("music-classical", "классическая музыка"), ("theatre-arts", "театр"),
    ("photography", "фотография"), ("painting", "живопись"), ("history", "история"), ("politics-local", "местная политика"),
    ("economy", "экономика"), ("startups", "стартапы"), ("marketing", "маркетинг"), ("hr-careers", "карьера и работа"),
    ("remote-work", "удалённая работа"), ("climate", "климат"), ("energy", "энергетика"), ("space", "космос"),
    ("science-bio", "биология"), ("science-physics", "физика"), ("weather", "погода"), ("public-transport", "общественный транспорт"),
    ("airlines-service", "сервис авиакомпаний"), ("hotels", "отели"), ("restaurants", "рестораны"), ("delivery-service", "доставка"),
    ("banking-service", "банковский сервис"), ("telecom", "мобильная связь"), ("internet-provider", "интернет-провайдер"),
    ("utilities", "коммунальные услуги"), ("e-commerce", "интернет-магазины"), ("marketplaces", "маркетплейсы"),
    ("insurance-claims", "страховые случаи"), ("legal-consumer", "права потребителей"), ("taxes", "налоги"),
    ("wedding", "свадьбы"), ("holidays", "праздники"), ("religion-culture", "культура и традиции"), ("volunteering", "волонтёрство"),
    ("city-life", "городская жизнь"), ("rural-life", "жизнь в деревне"), ("fishing", "рыбалка"), ("hunting-law", "охота и закон"),
    ("chess", "шахматы"), ("esports", "киберспорт"), ("anime", "аниме"), ("comics", "комиксы"), ("podcasts", "подкасты"),
]]
CONTENT_HOLDOUT: Pool = [
    ("beekeeping", "пчеловодство"), ("sailing", "парусный спорт"), ("opera", "опера"), ("archaeology", "археология"),
    ("knitting", "вязание"), ("aquariums", "аквариумистика"), ("mountaineering", "альпинизм"), ("perfume", "парфюмерия"),
    ("vinyl", "виниловые пластинки"), ("birdwatching", "наблюдение за птицами"), ("ceramics", "керамика"),
    ("figure-skating", "фигурное катание"), ("cheese", "сыроварение"), ("watches", "часы"), ("genealogy", "генеалогия"),
]

EXTRACT_TRAIN: Pool = [(c, d) for c, d in [
    ("paths", "пути и имена файлов"), ("commands", "команды оболочки"), ("versions", "версии ПО"), ("dates", "даты и время"),
    ("money", "суммы и валюты"), ("order-ids", "номера заказов/тикетов"), ("person-names", "имена людей (вымышленные)"),
    ("orgs", "организации"), ("addresses", "адреса (вымышленные)"), ("emails", "e-mail (example.com)"), ("urls", "URL"),
    ("hosts-ips", "хосты и IP (зарезервированные диапазоны)"), ("env-vars", "переменные окружения"), ("error-codes", "коды ошибок"),
    ("hashes", "хеши и идентификаторы коммитов"), ("config-params", "параметры конфигурации"), ("quantities", "числа с единицами"),
    ("roles", "роли и должности"), ("deadlines", "сроки"), ("conditions", "условия договора"), ("products", "названия товаров"),
    ("phone-fake", "телефоны (вымышленные, 555-xxx)"), ("percentages", "проценты и ставки"), ("languages-tech", "языки и технологии"),
]]
EXTRACT_HOLDOUT: Pool = [
    ("uuid", "UUID"), ("mac", "MAC-адреса"), ("hex-color", "цвета в hex"), ("isbn", "ISBN"), ("coords", "координаты"),
    ("chem-formula", "химические формулы"), ("music-chords", "аккорды"),
]

TRAIN: Dict[str, Pool] = {"R001": SAFETY_TRAIN, "R002": TOOLS_TRAIN, "R003": CONTENT_TRAIN, "R004": EXTRACT_TRAIN}
HOLDOUT: Dict[str, Pool] = {"R001": SAFETY_HOLDOUT, "R002": TOOLS_HOLDOUT, "R003": CONTENT_HOLDOUT, "R004": EXTRACT_HOLDOUT}
HOLDOUT_CODES = {code for pool in HOLDOUT.values() for code, _ in pool}


def check_pools() -> None:
    """Train and hold-out families never share a code (a shared code would put an exam family into training)."""
    for req in TRAIN:
        train = {c for c, _ in TRAIN[req]}
        hold = {c for c, _ in HOLDOUT[req]}
        if train & hold:
            raise ValueError(f"{req}: families in both train and hold-out: {sorted(train & hold)}")
        if len(train) != len(TRAIN[req]):
            raise ValueError(f"{req}: duplicate train family codes")
