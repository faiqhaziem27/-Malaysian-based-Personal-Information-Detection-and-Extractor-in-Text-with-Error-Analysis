import argparse
import json
import random
try:
    from project_paths import DATA_DIR, REQUIRED_SPLITS
    from label_config import LABEL_LIST
except ModuleNotFoundError:
    from src.project_paths import DATA_DIR, REQUIRED_SPLITS
    from src.label_config import LABEL_LIST

# Define the extended Malaysian raw dataset file path
MALAYSIA_EXTENDED_RAW_DATA_PATH = DATA_DIR / "malaysia" / "malaysia_ner_extended.json"

# Define the default number of samples and random seed
DEFAULT_NUM_SAMPLES = 100
DEFAULT_SEED = 2026

# Define the share of negative (all O) and single-entity samples
NEGATIVE_RATE = 0.12
SOLO_RATE = 0.08

# Define the template pool used by each split
# Validation and test use held-out templates that never appear in training
SPLIT_TEMPLATE_POOL = {
    "train": "train",
    "validation": "held",
    "test": "held"
}

# Define the sample ID prefix for each split
SPLIT_ID_PREFIX = {
    "train": "ext_train",
    "validation": "ext_val",
    "test": "ext_test"
}

# Define Malaysian name parts (combined to create many unique names)
MALAY_M = ["Adib", "Afiq", "Aqil", "Arif", "Azri", "Fahmi", "Faris", "Hadi", "Haikal",
           "Hazim", "Ikmal", "Ilyas", "Imran", "Izzat", "Johan", "Najmi", "Naufal", "Rafiq",
           "Rizal", "Shafiq", "Syahmi", "Taufiq", "Umar", "Zarif", "Zikri", "Amsyar", "Badrul"]
MALAY_F = ["Aleesya", "Batrisya", "Damia", "Elisya", "Fatin", "Hana", "Iman", "Irdina",
           "Khadijah", "Liyana", "Maisarah", "Nabila", "Natasha", "Qistina", "Raihana",
           "Safiya", "Syamimi", "Umairah", "Yasmin", "Zahra", "Athirah", "Anis", "Dhia"]
MALAY_FATHER = ["Azhar", "Baharudin", "Daud", "Ghazali", "Hashim", "Idris", "Jaafar",
                "Kassim", "Latif", "Mansor", "Nordin", "Rosli", "Sulaiman", "Tajuddin",
                "Wahab", "Yaacob", "Zakaria", "Harun", "Musa", "Shamsudin"]
MALAY_PREFIX_M = ["", "", "", "", "Wan", "Nik", "Megat", "Tengku", "Syed", "Raja"]
MALAY_PREFIX_F = ["", "", "", "", "Wan", "Nik", "Tengku", "Sharifah", "Puteri", "Raja"]

# Define Chinese name parts
CN_SURNAME = ["Chua", "Ho", "Kang", "Loh", "Low", "Mak", "Pang", "Quek", "Seow", "Soh",
              "Tay", "Tee", "Toh", "Wee", "Yeoh", "Yong", "Choo", "Gan", "Heng", "Leong"]
CN_GIVEN = ["Ai", "Bee", "Choon", "Fei", "Hong", "Jun", "Kai", "Keat", "Lian", "Mun",
            "Pei", "Qian", "Rui", "Sheng", "Shu", "Tian", "Wen", "Xuan", "Yen", "Zi", "Yu"]
CN_ENGLISH = ["Brandon", "Cheryl", "Desmond", "Eunice", "Gavin", "Isaac", "Janice",
              "Marcus", "Nicole", "Samantha", "Terence", "Valerie"]

# Define Indian and Sikh name parts
IN_M = ["Aravind", "Dinesh", "Gopal", "Hari", "Jeevan", "Karthik", "Logesh", "Mugilan",
        "Pravin", "Saravanan", "Thinesh", "Yuvaraj", "Kishore", "Tharshan"]
IN_F = ["Abirami", "Bavani", "Durga", "Gayathri", "Janani", "Kalaivani", "Lavanya",
        "Nithya", "Pavithra", "Sharmila", "Tamilarasi", "Vasanthi", "Yamini"]
IN_FATHER = ["Arumugam", "Balakrishnan", "Chandran", "Govindasamy", "Kandasamy",
             "Maniam", "Narayanan", "Perumal", "Ramalingam", "Sivakumar", "Velu"]
SIKH_FIRST = ["Gurpreet", "Jasvinder", "Manjit", "Parveen", "Sukhdev", "Amarjit"]  # unisex in use

# Define Sabah and Sarawak names
BORNEO = [["Alvin", "anak", "Liang"], ["Melissa", "anak", "Sait"], ["Jonathan", "anak", "Ngau"],
          ["Grace", "anak", "Bujang"], ["Rodney", "Majalang"], ["Florence", "Gimbad"],
          ["Ezekiel", "Gaban"], ["Juliana", "Lajim"], ["Benedict", "Ganang"]]

# Define titles that match the person's gender
# Professional titles (Dr., Prof., Encik) are tagged O, but Malaysian honorifics
# (Dato', Datin) are part of the person span, following malaysia_ner_samples.json
HONORIFIC_TITLES = {"Dato'", "Datin"}
TITLES_M = ["Dr.", "Encik", "Tuan", "Prof.", "Dato'", "Ir.", "Ts.", "Mr."]
TITLES_F = ["Dr.", "Puan", "Cik", "Prof.", "Datin", "Ir.", "Ts.", "Ms."]

# Define organisations as (full name, abbreviation, locations)
# Companies also have a type so each template only uses companies that fit it
# None of these appear in malaysia_ner_samples.json
ORGS = {
    "uni": [
        ("Universiti Tun Hussein Onn Malaysia", "UTHM", ["Parit Raja", "Batu Pahat"]),
        ("Universiti Malaysia Perlis", "UniMAP", ["Arau", "Kangar"]),
        ("Universiti Teknikal Malaysia Melaka", "UTeM", ["Durian Tunggal"]),
        ("Universiti Malaysia Kelantan", "UMK", ["Bachok", "Jeli"]),
        ("Universiti Sains Islam Malaysia", "USIM", ["Nilai"]),
        ("Universiti Pertahanan Nasional Malaysia", "UPNM", ["Sungai Besi"]),
        ("Universiti Tenaga Nasional", "UNITEN", ["Bandar Muadzam Shah"]),
        ("Universiti Sultan Zainal Abidin", "UniSZA", ["Gong Badak"]),
        ("Universiti Malaysia Pahang Al-Sultan Abdullah", "UMPSA", ["Pekan", "Gambang"]),
        ("Universiti Kuala Lumpur", "UniKL", ["Alor Gajah", "Lumut"]),
        ("Heriot-Watt University Malaysia", None, ["Precinct 5"]),
        ("University of Nottingham Malaysia", None, ["Semenyih"]),
        ("Xiamen University Malaysia", None, ["Salak Tinggi"]),
        ("INTI International University", None, ["Nilai"]),
        ("SEGi University", None, ["Kota Damansara"]),
        ("Management and Science University", "MSU", ["Seksyen 13"]),
        ("Politeknik Port Dickson", None, ["Port Dickson"]),
    ],
    "hosp": [
        ("Hospital Kemaman", None, ["Kemaman"]),
        ("Hospital Selayang", None, ["Selayang"]),
        ("Hospital Sungai Buloh", None, ["Sungai Buloh"]),
        ("Hospital Melaka", None, ["Bandar Melaka"]),
        ("Hospital Sultan Abdul Halim", "HSAH", ["Sungai Petani"]),
        ("Hospital Seberang Jaya", None, ["Seberang Jaya"]),
        ("Hospital Taiping", None, ["Taiping"]),
        ("Hospital Teluk Intan", None, ["Teluk Intan"]),
        ("Hospital Sibu", None, ["Sibu"]),
        ("Hospital Tawau", None, ["Tawau"]),
        ("KPJ Kluang Utama Specialist Hospital", None, ["Kluang"]),
        ("Thomson Hospital Kota Damansara", None, ["Kota Damansara"]),
        ("Hospital Muar", None, ["Muar"]),
        ("ParkCity Medical Centre", None, ["Desa ParkCity"]),
        ("Hospital Sultan Haji Ahmad Shah", None, ["Temerloh"]),
        ("Klinik Kesihatan Batu Pahat", None, ["Batu Pahat"]),
        ("Hospital Enche' Besar Hajjah Khalsom", None, ["Kluang"]),
    ],
    "co": [
        ("Bank Islam Malaysia", "Bank Islam", ["Pasir Mas", "Tanah Merah"], "bank"),
        ("Bank Rakyat", None, ["Raub", "Bentong"], "bank"),
        ("Bank Simpanan Nasional", "BSN", ["Rembau", "Nilai"], "bank"),
        ("Affin Bank", None, ["Taiping", "Lumut"], "bank"),
        ("Alliance Bank", None, ["Kulai", "Pontian"], "bank"),
        ("U Mobile", None, ["Bukit Mertajam"], "telco"),
        ("Shopee Malaysia", "Shopee", ["Rawang"], "ecommerce"),
        ("Lazada Malaysia", "Lazada", ["Setia Alam"], "ecommerce"),
        ("Lalamove", None, ["Butterworth", "Seri Kembangan"], "courier"),
        ("Ninja Van Malaysia", "Ninja Van", ["Senai", "Rawang"], "courier"),
        ("KTM Berhad", "KTMB", ["Gemas", "Butterworth"], "transport"),
        ("Indah Water Konsortium", "IWK", ["Muar", "Kemaman"], "sewerage"),
        ("Jabatan Pengangkutan Jalan", "JPJ", ["Wangsa Maju", "Kuala Kangsar"], "government"),
        ("Jabatan Pendaftaran Negara", "JPN", ["Bukit Mertajam", "Sungai Petani"], "government"),
        ("Perbadanan Tabung Pendidikan Tinggi Nasional", "PTPTN", ["Batu Pahat", "Sibu"], "loan"),
        ("Takaful Malaysia", None, ["Bintulu"], "insurance"),
        ("Mydin", None, ["Seri Kembangan", "Gong Badak", "Bukit Mertajam"], "retail"),
        ("AEON Malaysia", "AEON", ["Bukit Tinggi", "Bukit Indah"], "retail"),
        ("99 Speedmart", None, ["Banting", "Kluang"], "retail"),
        ("Sarawak Energy", None, ["Sibu", "Bintulu"], "electricity"),
        ("Sabah Electricity", None, ["Keningau", "Lahad Datu"], "electricity"),
        ("Majlis Bandaraya Iskandar Puteri", "MBIP", ["Iskandar Puteri"], "government"),
    ],
}

# Define extra locations for sentences without an organisation
EXTRA_LOC = ["Muar", "Segamat", "Kulai", "Pasir Gudang", "Teluk Intan", "Temerloh",
             "Bentong", "Kemaman", "Dungun", "Port Dickson", "Labuan", "Bintulu",
             "Rawang", "Setia Alam", "Kangar", "Langkawi", "Cameron Highlands",
             "Tanah Merah", "Pontian", "Raub", "Lumut", "Sri Aman"]

# Define sentence templates for each domain
# Slots: {PER} full name, {PER2} second person, {FN} first name only,
# {T} title, {ORG} organisation, {LOC} location
# Company templates are (template, allowed company types); None allows any type
TEMPLATES = {
    "uni": {
        "train": [
            "Saya {PER} , pelajar tahun dua di {ORG} , tidak dapat log masuk ke portal pelajar .",
            "Hi admin , {FN} here from {ORG} , the e-learning page cannot load since morning .",
            "{T} {PER} has been appointed as the new deputy dean at {ORG} .",
            "Kindly approve the hostel transfer for {PER} from {ORG} {LOC} campus .",
            "Pihak {ORG} di {LOC} memaklumkan bahawa {PER} telah menerima biasiswa .",
            "Eh bro , the wifi at {ORG} {LOC} down again lah .",
            "Dear {T} {PER} , your thesis submission at {ORG} has been received .",
            "{PER} and {PER2} represented {ORG} at the hackathon in {LOC} .",
            "Name : {PER} | Institution : {ORG} | Campus : {LOC}",
            "The library card issued to {PER} by {ORG} has expired .",
            "Kursus pendek anjuran {ORG} akan diadakan di {LOC} minggu depan .",
            "Can someone help {FN} ? The course registration at {ORG} keeps showing an error .",
        ],
        "held": [
            "Permohonan penangguhan pengajian oleh {PER} sedang diproses oleh {ORG} .",
            "Final year student {PER} reported a broken lab PC at {ORG} in {LOC} .",
            "Sorry ah , {FN} from {ORG} , I cannot download my exam slip .",
            "Congratulations to {PER} of {ORG} for winning the innovation award in {LOC} .",
        ],
    },
    "hosp": {
        "train": [
            "Pesakit atas nama {PER} perlu hadir ke {ORG} untuk temujanji susulan .",
            "Hi , I want to ask whether {ORG} in {LOC} still opens the pharmacy on Sunday ?",
            "{T} {PER} from {ORG} will review the X-ray results tomorrow .",
            "My mother {PER} was transferred from {ORG} to another ward in {LOC} .",
            "Appointment confirmed : {PER} , Klinik Pakar Ortopedik , {ORG} .",
            "Staff nurse {PER} and {PER2} handled the night shift at {ORG} .",
            "Kenapa sistem giliran di {ORG} {LOC} lambat sangat hari ini ?",
            "Please send the medical leave certificate for {PER} to {ORG} .",
            "Patient name : {PER} | Hospital : {ORG} | District : {LOC}",
            "{ORG} in {LOC} reported that its online booking system is back to normal .",
        ],
        "held": [
            "Waris kepada {PER} diminta menghubungi {ORG} dengan segera .",
            "Can I reschedule the check-up for {PER} at {ORG} to next week ?",
            "The emergency department at {ORG} in {LOC} is full tonight , says {T} {PER} .",
        ],
    },
    "co": {
        "train": [
            ("Saya {PER} ingin membuat aduan tentang caj berganda oleh {ORG} .", ["bank", "telco", "ecommerce", "electricity", "insurance"]),
            ("Hello {ORG} , my parcel stuck at the {LOC} hub for one week already .", ["courier", "ecommerce"]),
            ("{PER} has been promoted to regional manager at {ORG} in {LOC} .", ["bank", "telco", "ecommerce", "courier", "retail", "insurance", "electricity"]),
            ("Dear customer {PER} , your {ORG} account statement is now available .", ["bank", "telco", "insurance", "loan"]),
            ("Why {ORG} app keep logging me out ah ? - {FN}", ["bank", "telco", "ecommerce", "courier", "retail", "transport"]),
            ("{T} {PER} signed the partnership agreement on behalf of {ORG} in {LOC} .", ["bank", "telco", "ecommerce", "courier", "retail", "insurance", "electricity", "transport", "sewerage"]),
            ("Account holder : {PER} | Bank : {ORG} | Branch : {LOC}", ["bank"]),
            ("Rider {PER} from {ORG} could not find the address in {LOC} .", ["courier"]),
            ("Bil elektrik daripada {ORG} untuk rumah di {LOC} terlalu tinggi bulan ini .", ["electricity"]),
            ("{PER} and {PER2} from {ORG} attended the career fair in {LOC} .", None),
            ("The {ORG} outlet in {LOC} will close early today for stock taking .", ["retail", "telco"]),
            ("Tolong semak status bayaran balik {ORG} atas nama {PER} .", ["loan", "bank", "ecommerce", "insurance"]),
        ],
        "held": [
            ("Wakil {ORG} , {PER} , akan menghubungi anda dalam masa tiga hari bekerja .", None),
            ("I already called {ORG} three times but nobody answer , very frustrating - {FN}", None),
            ("A complaint from {PER} about the {ORG} branch in {LOC} has been escalated .", ["bank", "retail", "telco", "government", "insurance"]),
            ("Courier from {ORG} left the package at the guardhouse in {LOC} .", ["courier", "ecommerce"]),
        ],
    },
}

# Define location-only and person-only templates
# These stop the model from expecting every sentence to have PER, ORG and LOC
SOLO = {
    "train": [
        "Banjir kilat dilaporkan di {LOC} petang tadi .",
        "Traffic jam all the way from {LOC} to {LOC2} this morning .",
        "{PER} will be on leave until next Monday .",
        "Please call {T} {PER} for further information .",
        "Road closure in {LOC} due to the marathon on Sunday .",
    ],
    "held": [
        "Bekalan air terputus di kawasan {LOC} sehingga esok .",
        "Thank you {PER} for your quick response .",
    ],
}

# Define negative sentences with no entities
NEGATIVES = {
    "train": [
        "Sistem sedang diselenggara , sila cuba sebentar lagi .",
        "Your request has been received and will be processed shortly .",
        "Boleh tak tolong check balik ?",
        "The password must contain at least eight characters .",
        "Terima kasih kerana menghubungi kami .",
        "Ok noted , I will update you later .",
        "Please do not share your one-time password with anyone .",
        "Server down again , so annoying lah .",
        "Kindly fill in the form before Friday .",
        "Sila kemas kini maklumat peribadi anda .",
        "The meeting has been moved to the afternoon .",
        "Nak tanya , bila result keluar ?",
        "The link in the email has expired .",
        "Saya tak terima kod pengesahan lagi .",
        "Please upload the document in PDF format .",
        "Can I change my delivery address ?",
        "Bayaran anda telah diterima .",
        "The queue number is not moving at all .",
        "Our office is closed on public holidays .",
        "Boleh saya tahu kenapa akaun saya disekat ?",
        "Please wait while we transfer you to the next available agent .",
        "Aplikasi asyik crash bila buka .",
        "I have attached the screenshot for your reference .",
        "Kindly ignore this message if you have already made payment .",
    ],
    "held": [
        "Maaf atas kesulitan yang dialami .",
        "We have escalated the matter to the relevant team .",
        "Sorry for the late reply , very busy today .",
        "Permohonan anda telah berjaya dihantar .",
        "The page shows a blank screen after login .",
        "Boleh tolong reset kata laluan saya ?",
        "Your appointment slot is no longer available .",
        "Kami sedang menyiasat masalah ini .",
    ],
}


# Define greetings and closings added to negative sentences
NEG_OPEN = ["", "", "Hi ,", "Hello ,", "Salam ,", "Dear customer ,", "Assalamualaikum ,",
            "Hai ,", "Good morning ,", "Excuse me ,"]
NEG_CLOSE = ["", "", "Thank you .", "Terima kasih .", "TQ .", "Sorry ya .", "Please advise .",
             "Harap maklum ."]



def generate_person(rng):
    """
    Generate a random Malaysian person name.

    Returns:
        tuple: (name tokens, gender) where gender is "M", "F" or None.
    """
    name_type = rng.random()
    is_male = rng.random() < 0.5
    gender = "M" if is_male else "F"

    # Malay name, for example "Wan Aqil bin Hashim"
    if name_type < 0.45:
        given_names = rng.sample(MALAY_M if is_male else MALAY_F, rng.choice([1, 1, 2]))
        prefix = rng.choice(MALAY_PREFIX_M if is_male else MALAY_PREFIX_F)
        name = ([prefix] if prefix else []) + given_names

        if rng.random() < 0.6:
            name += ["bin" if is_male else "binti", rng.choice(MALAY_FATHER)]
        elif rng.random() < 0.5:
            name += [rng.choice(MALAY_FATHER)]

        return name, gender

    # Chinese name, for example "Yeoh Wen Jun" or "Marcus Yeoh"
    if name_type < 0.70:
        name = [rng.choice(CN_SURNAME)] + rng.sample(CN_GIVEN, 2)

        if rng.random() < 0.3:
            name = [rng.choice(CN_ENGLISH)] + name
        elif rng.random() < 0.15:
            name = [rng.choice(CN_ENGLISH), name[0]]

        return name, None

    # Indian name, for example "Dinesh a/l Perumal"
    if name_type < 0.88:
        name = [rng.choice(IN_M if is_male else IN_F)]

        if rng.random() < 0.5:
            name += ["a/l" if is_male else "a/p"]

        return name + [rng.choice(IN_FATHER)], gender

    # Sikh name, for example "Gurpreet Kaur"
    if name_type < 0.94:
        return [rng.choice(SIKH_FIRST), "Singh" if is_male else "Kaur"], gender

    # Sabah or Sarawak name, for example "Alvin anak Liang"
    return list(rng.choice(BORNEO)), None


def generate_title(rng, gender):
    """
    Pick a title that matches the person's gender.

    Returns:
        str: The title, for example "Encik" or "Dr.".
    """
    if gender == "M":
        return rng.choice(TITLES_M)

    if gender == "F":
        return rng.choice(TITLES_F)

    return rng.choice(["Dr.", "Prof.", "Ir.", "Ts."])


def generate_first_name(rng):
    """
    Generate a short first name used in chat messages.

    Returns:
        list: A single name token, for example ["Aqil"].
    """
    return [rng.choice(MALAY_M + MALAY_F + CN_ENGLISH + IN_M + IN_F)]


def generate_negative_sentence(rng, template_pool):
    """
    Generate a negative sentence with an optional greeting and closing.

    Returns:
        str: The sentence with tokens separated by spaces.
    """
    sentence_body = rng.choice(NEGATIVES[template_pool])
    sentence_parts = (rng.choice(NEG_OPEN), sentence_body, rng.choice(NEG_CLOSE))

    # Lowercase the first letter when a greeting comes before the sentence
    if sentence_parts[0]:
        sentence_parts = (sentence_parts[0], sentence_body[0].lower() + sentence_body[1:], sentence_parts[2])

    return " ".join(part for part in sentence_parts if part)


def pick_organisation(rng, domain, allowed_types=None):
    """
    Pick an organisation and one of its locations.

    Returns:
        tuple: (organisation name, location name).
    """
    candidates = [
        organisation
        for organisation in ORGS[domain]
        if allowed_types is None or organisation[3] in allowed_types
    ]
    full_name, abbreviation, locations = rng.choice(candidates)[:3]

    # Use the abbreviation for about a third of the samples
    if abbreviation and rng.random() < 0.35:
        organisation = abbreviation
    else:
        organisation = full_name

    return organisation, rng.choice(locations)


def fill_template(template, rng, domain=None, allowed_types=None):
    """
    Fill a template with random entities and create BIO labels.

    Returns:
        tuple: (tokens, ner_tags) with the same length.
    """
    if domain:
        organisation, organisation_location = pick_organisation(rng, domain, allowed_types)
    else:
        organisation, organisation_location = None, None

    # Generate two different people for templates with {PER2}
    (person, gender), (second_person, _) = generate_person(rng), generate_person(rng)
    while second_person == person:
        second_person, _ = generate_person(rng)

    slot_values = {
        "{PER}": (person, "PER"),
        "{PER2}": (second_person, "PER"),
        "{FN}": (generate_first_name(rng), "PER"),
        "{ORG}": (organisation.split() if organisation else [], "ORG"),
        "{LOC}": ((organisation_location or rng.choice(EXTRA_LOC)).split(), "LOC"),
        "{LOC2}": (rng.choice(EXTRA_LOC).split(), "LOC")
    }

    tokens = []
    ner_tags = []
    honorific_before_person = False

    for part in template.split():
        if part in slot_values:
            entity_words, entity_type = slot_values[part]

            for index, word in enumerate(entity_words):
                tokens.append(word)

                # Continue the person span when an honorific starts it
                if index == 0 and not honorific_before_person:
                    ner_tags.append("B-" + entity_type)
                else:
                    ner_tags.append("I-" + entity_type)

            honorific_before_person = False

        elif part == "{T}":
            title = generate_title(rng, gender)
            tokens.append(title)

            if title in HONORIFIC_TITLES:
                ner_tags.append("B-PER")
                honorific_before_person = True
            else:
                ner_tags.append("O")

        else:
            tokens.append(part)
            ner_tags.append("O")

    return tokens, ner_tags


def generate_sample(rng, template_pool):
    """
    Generate one sample: a negative, a single-entity or a full sentence.

    Returns:
        tuple: (tokens, ner_tags) with the same length.
    """
    sample_type = rng.random()

    if sample_type < NEGATIVE_RATE:
        tokens = generate_negative_sentence(rng, template_pool).split()
        return tokens, ["O"] * len(tokens)

    if sample_type < NEGATIVE_RATE + SOLO_RATE:
        return fill_template(rng.choice(SOLO[template_pool]), rng)

    domain = rng.choice(list(TEMPLATES))
    template = rng.choice(TEMPLATES[domain][template_pool])

    # Company templates come with the company types they allow
    if isinstance(template, tuple):
        template, allowed_types = template
    else:
        allowed_types = None

    return fill_template(template, rng, domain, allowed_types)


def build_dataset(num_samples, seed):
    """
    Build the extended Malaysian NER dataset with a 70/15/15 split.

    Returns:
        dict: Contains train, validation, and test sample lists.
    """
    rng = random.Random(seed)

    # Define the number of samples in each split
    split_sizes = {
        "train": int(num_samples * 0.7),
        "validation": int(num_samples * 0.15)
    }
    split_sizes["test"] = num_samples - split_sizes["train"] - split_sizes["validation"]

    seen_sentences = set()
    dataset = {}

    for split in REQUIRED_SPLITS:
        split_samples = []
        attempts = 0

        while len(split_samples) < split_sizes[split] and attempts < split_sizes[split] * 100:
            attempts += 1
            tokens, ner_tags = generate_sample(rng, SPLIT_TEMPLATE_POOL[split])

            # Skip duplicate sentences
            sentence = " ".join(tokens)
            if sentence in seen_sentences:
                continue
            seen_sentences.add(sentence)

            assert len(tokens) == len(ner_tags), (
                f"Token-label length mismatch in {split}: {sentence}"
            )
            assert all(tag in LABEL_LIST for tag in ner_tags), (
                f"Unknown label in {split}: {sentence}"
            )

            split_samples.append({
                "id": f"{SPLIT_ID_PREFIX[split]}_{len(split_samples) + 1:03d}",
                "tokens": tokens,
                "ner_tags": ner_tags
            })

        dataset[split] = split_samples

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate the extended Malaysian NER dataset.")
    parser.add_argument("--n", type=int, default=DEFAULT_NUM_SAMPLES, help="Total number of samples")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--out", default=str(MALAYSIA_EXTENDED_RAW_DATA_PATH), help="Output JSON path")
    args = parser.parse_args()

    # Build the extended Malaysian dataset
    dataset = build_dataset(args.n, args.seed)

    # Save the dataset in the same format as malaysia_ner_samples.json
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=2)

    # Display the extended Malaysian dataset split sizes
    print("\nExtended Malaysian Dataset Split Sizes:\n")

    for split in REQUIRED_SPLITS:
        print(f"{split.capitalize():<12}: {len(dataset[split])} samples")

    print(f"\nDataset saved to: {args.out}")


if __name__ == "__main__":
    main()
