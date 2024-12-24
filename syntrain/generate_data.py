import random
import pandas as pd
from tqdm import trange, tqdm
import json
import math
import os
import torch
import glob
import pickle
from multiprocessing import Pool
from transformers import LlamaTokenizer


last_names = None
male_names = None
female_names = None
middle_names = None
famous_universities = None
popular_majors = None
places = None
BirthYears = None
BirthMonths = None
BirthDays = None
birthplace_templates = None
birthdate_templates = None
university_templates = None
major_templates = None
workplace_templates = None
total_possible_names = 0
generated_name = set()

size2num = {
    "tiny": 10000,
    "small": 100000,
    "medium": 1000000,
    "large": 5000000,
}
def load_all_templates(path:str = "data/templates.json"):
    with open(path, "r", encoding='utf8') as f:
        templates = json.load(f)
    global birthplace_templates, birthdate_templates, university_templates, major_templates, workplace_templates
    # birthplace_templates = templates["birthplace_templates"]
    # birthdate_templates = templates["birthdate_templates"]
    # university_templates = templates["university_templates"]
    # major_templates = templates["major_templates"]
    # workplace_templates = templates["workplace_templates"]
    birthplace_templates = ["[Name] was born in [Birthplace]."]
    birthdate_templates = ["[Name] met the world on [Birthdate]."]
    university_templates = ["[Name] studied at [University]."]
    major_templates = ["[Name] was a [Major] student."]
    workplace_templates = ["[Name] left for [Workplace] to get a job."]
    print(f"Birthplace templates Num: {len(birthplace_templates)}")
    print(f"Birthdate templates Num: {len(birthdate_templates)}")
    print(f"University templates Num: {len(university_templates)}")
    print(f"Major templates Num: {len(major_templates)}")
    print(f"Workplace templates Num: {len(workplace_templates)}")
def load_all_possible_features(path:str = "data/all_possible_features.json"):
    with open(path, "r", encoding='utf8') as f:
        features = json.load(f)
    global last_names, male_names, female_names, middle_names, \
    famous_universities, popular_majors, places, BirthYears, BirthMonths, BirthDays
    last_names = features["last_names"]
    male_names = features["male_first_names"]
    female_names = features["female_first_names"]
    # middle_names = features["middle_names"]
    middle_names = ["A", "B", "C", "D", "E", "F", "G", 
                    "H", "I", "J", "K", "L", "M", "N", 
                    "O", "P", "Q", "R", "S", "T", 
                    "U", "V", "W", "X", "Y", "Z"]
    famous_universities = features["famous_universities"]
    popular_majors = features["popular_majors"]
    places = features["places"]
    BirthYears = features["birth_years"]
    BirthMonths = features["birth_months"]
    BirthDays = features["birth_days"]
    all_possible_names = (len(male_names)+len(female_names)) * len(middle_names) * len(last_names)
    all_possible_birthdays = len(BirthYears) * len(BirthMonths) * len(BirthDays)
    global total_possible_names
    total_possible_names = all_possible_names
    print(f"{len(male_names)+len(female_names)}*{len(middle_names)}*{len(last_names)}={all_possible_names} possible names.")
    print(f"{len(BirthYears)}*{len(BirthMonths)}*{len(BirthDays)}={all_possible_birthdays} possible birthdays.")
    print(f"{len(famous_universities)} possible universities.")
    print(f"{len(popular_majors)} possible majors.")
    print(f"{len(places)} possible workplaces/birthplaces.")

def augtimes2num(n):
    # 2 ~ 101: the porbability follows exponential distribution
    augtimes2rank = {}
    for i in range(1, 101):
        prob = 1 - math.exp(-i/25) # \lambda = 25
        augtimes2rank[i + 1] = int(n * prob)
    return augtimes2rank
def generate_one_profile(augment_times):
    last_name = random.choice(last_names)
    middle_name = random.choice(middle_names)
    gender = random.choice(["Male", "Female"])
    if gender == 'Male':
        first_name = random.choice(male_names)
    else:
        first_name = random.choice(female_names)
    full_name = first_name + " " + middle_name + " " + last_name
    if full_name in generated_name: # sample again if dumplicated
        return generate_one_profile(augment_times=augment_times)
    generated_name.add(full_name)
    birthday = f"{random.choice(BirthYears)}-{random.choice(BirthMonths)}-{random.choice(BirthDays)}"
    birthplace = random.choice(places)
    university = random.choice(famous_universities)
    major = random.choice(popular_majors)
    work_place = random.choice(places)

    info = {
        "Fullname": full_name,
        "Augmenttimes": augment_times,
        "Birthdate": birthday,
        "Birthplace": birthplace,
        "University": university,
        "Major": major,
        "Workplace": work_place
    }
    return info
def generate_profiles(num, root_path, is_balanced=True):
    """ Generate profiles for random fake people """
    if os.path.exists(root_path + "people_profiles.parquet"):
        profiles = pd.read_parquet(root_path + "people_profiles.parquet")
        print(f"Load {len(profiles)} profiles in total.")
    else:
        augtime2rank = augtimes2num(num) # used when is_balanced is False
        profiles = []
        current_augment_times = 2 # the minimum augment times, one for train and one for valid
        for i in trange(num):
            person_info = generate_one_profile(augment_times=current_augment_times)
            profiles.append(person_info)
            if not is_balanced:
                if current_augment_times <= 101 and i >= augtime2rank[current_augment_times]:
                    current_augment_times += 1
                if current_augment_times > 101:
                    break
        profiles = pd.DataFrame(profiles)
        profiles.to_parquet(root_path + "people_profiles.parquet")
        print(f"Generate {len(profiles)} profiles in total.")
    return profiles

def create_a_person_biography(info, is_balanced=True):
    # generate biographies according to the fake person info
    augment_times = info["Augmenttimes"]
    biographies = []
    for i in range(augment_times):
        sentences = []
        birthplace_template = random.choice(birthplace_templates)
        birthplace_template = birthplace_template.replace("[Name]", info["Fullname"])
        birthplace_template = birthplace_template.replace("[Birthplace]", info["Birthplace"])
        sentences.append(birthplace_template)
        birthdate_template = random.choice(birthdate_templates)
        birthdate_template = birthdate_template.replace("[Name]", info["Fullname"])
        birthdate_template = birthdate_template.replace("[Birthdate]", info["Birthdate"])
        sentences.append(birthdate_template)
        university_template = random.choice(university_templates)
        university_template = university_template.replace("[Name]", info["Fullname"])
        university_template = university_template.replace("[University]", info["University"])
        sentences.append(university_template)
        major_template = random.choice(major_templates)
        major_template = major_template.replace("[Name]", info["Fullname"])
        major_template = major_template.replace("[Major]", info["Major"])
        sentences.append(major_template)
        workplace_template = random.choice(workplace_templates)
        workplace_template = workplace_template.replace("[Name]", info["Fullname"])
        workplace_template = workplace_template.replace("[Workplace]", info["Workplace"])
        sentences.append(workplace_template)
        if not is_balanced: # use permutation
            random.shuffle(sentences)
        biography = " ".join(sentences)
        biographies.append(biography)
    return biographies
def generate_biographies(num, root_path, profiles, is_balanced=True):
    """ Generate biographies for the profiles """
    if os.path.exists(root_path + "train_bios.parquet") and os.path.exists(root_path + "test_bios.parquet"):
        train_biographies = pd.read_parquet(root_path + "train_bios.parquet")['content'].to_list()
        test_biographies = pd.read_parquet(root_path + "test_bios.parquet")['content'].to_list()
        print(f"Load training biographies Num: {len(train_biographies)}")
    else:
        train_biographies = []
        test_biographies = []
        for i in trange(len(profiles)):
            info = profiles.iloc[i].to_dict()
            biographies = create_a_person_biography(info, is_balanced)
            train_biographies.extend(biographies[:-1]) # last one for valid
            test_biographies.append(biographies[-1])
        pd.DataFrame({"content": train_biographies}).to_parquet(root_path + "train_bios.parquet")
        pd.DataFrame({"content": test_biographies}).to_parquet(root_path + "test_bios.parquet")
        print(f"Generate training biographies Num: {len(train_biographies)}")
    random.shuffle(train_biographies) # shuffle the training biographies
    return train_biographies

def create_a_person_qa(info):
    full_name = info["Fullname"]
    rag_info = {
        "Fullname": full_name,
        "Augmenttimes": info["Augmenttimes"],
        "is_train": False,
        "Q_bd": f"What is the birth date of {full_name}?",
        "Q_bp": f"What is the birth city of {full_name}?",
        "Q_uni": f"What is the university of {full_name}?",
        "Q_major": f"What major did {full_name} study?",
        "Q_wp": f"Where does {full_name} work?",
        "A_bd": info["Birthdate"],
        "A_bp": info["Birthplace"],
        "A_uni": info["University"],
        "A_major": info["Major"],
        "A_wp": info["Workplace"],
    }
    # info['augment_times'] = 5
    # contexts_right = create_a_person_biography(info)
    # for i in range(5):
    #     rag_info["factual_context_{}".format(i)] = contexts_right[i]

    # for i in range(3):
    #     birthday = info['birthday']
    #     while birthday == info['birthday']:
    #         birthday = f"{random.choice(BirthYears)}-{random.choice(BirthMonths)}-{random.choice(BirthDays)}"
    #     birthplace = info['birthplace']
    #     while birthplace == info['birthplace']:
    #         birthplace = random.choice(work_places)
    #     university = info['university']
    #     while university == info['university']:
    #         university = random.choice(famous_universities)
    #     major = info['major']
    #     while major == info['major']:
    #         major = random.choice(popular_majors)
    #     work_place = info['work_place']
    #     while work_place == info['work_place']:
    #         work_place = random.choice(work_places)
        
    #     counter_info = {
    #         "full_name": full_name,
    #         "gender": info["gender"],
    #         "birthday": birthday,
    #         "birthplace": birthplace,
    #         "university": university,
    #         "major": major,
    #         "work_place": work_place,
    #         "augment_times": 5
    #     }
    #     contexts_counter = create_a_person_biography(counter_info)
    #     for j in range(5):
    #         rag_info["counter{}_context_{}".format(i, j)] = contexts_counter[j]
    #     rag_info["counter{}_A_bd".format(i)] = birthday
    #     rag_info["counter{}_A_bp".format(i)] = birthplace
    #     rag_info["counter{}_A_uni".format(i)] = university
    #     rag_info["counter{}_A_major".format(i)] = major
    #     rag_info["counter{}_A_wp".format(i)] = work_place
    return rag_info
def generate_qa_profile(num, root_path, profiles):
    """ Generate qa data """
    if os.path.exists(root_path + "qa_profile.parquet"):
        qa_data = pd.read_parquet(root_path + "qa_profile.parquet")
        print(f"Load QA data Num: {len(qa_data)}")
    else:
        augtime2rank = augtimes2num(num)
        qa_data = []
        for i in trange(len(profiles)):
            info = profiles.iloc[i].to_dict()
            rag_info = create_a_person_qa(info)
            qa_data.append(rag_info)
        qa_data = pd.DataFrame(qa_data)
        # sample half for training(mixed-pretraining or finetuning), the other half for testing
        for key in tqdm(augtime2rank.keys()):
            augment_times2people = qa_data[qa_data['Augmenttimes'] == key]
            augment_times2people = augment_times2people.sample(frac=1) # shuffle
            train_people_num = int(len(augment_times2people) * 0.5)
            for i in range(train_people_num):
                qa_data.loc[qa_data["Fullname"] == augment_times2people.iloc[i]["Fullname"], "is_train"] = True
        qa_data.to_parquet(root_path + "qa_profile.parquet")
        print(f"Generate QA data Num: {len(qa_data) * 5}")

    if os.path.exists(root_path + "test_qas.parquet") and os.path.exists(root_path + "train_qas.parquet"):
        test_qa = pd.read_parquet(root_path + "test_qas.parquet")['content'].to_list()
        train_qa = pd.read_parquet(root_path + "train_qas.parquet")['content'].to_list()
        print(f"Load Training QA Num: {len(train_qa)}")
        print(f"Load Testing QA Num: {len(test_qa)}")
    else:
        test_people = qa_data[qa_data["is_train"] == False]
        test_qa = []
        for i in trange(len(test_people)):
            info = test_people.iloc[i].to_dict()
            for type_info in ["bd", "bp", "uni", "major", "wp"]:
                question = info[f"Q_{type_info}"]
                answer = info[f"A_{type_info}"]
                test_qa.append("Question:"+question+'\nAnswer:'+answer)
        pd.DataFrame({"content": test_qa}).to_parquet(root_path + "test_qas.parquet")
        train_people = qa_data[qa_data["is_train"] == True]
        train_qa = []
        for i in trange(len(train_people)):
            info = train_people.iloc[i].to_dict()
            for type_info in ["bd", "bp", "uni", "major", "wp"]:
                question = info[f"Q_{type_info}"]
                answer = info[f"A_{type_info}"]
                train_qa.append("Question:"+question+'\nAnswer:'+answer)
        pd.DataFrame({"content": train_qa}).to_parquet(root_path + "train_qas.parquet")
        print(f"Generate Training QA Num: {len(train_qa)}")
        print(f"Generate Testing QA Num: {len(test_qa)}")
    return qa_data, train_qa, test_qa

def construct_check_data(size, is_balanced):
    style = "balanced" if is_balanced else "augmented"
    if os.path.exists(f"data/{size}_{style}/check_data.jsonl"):
        print(f"Check data already exists.")
        return
    pretrain_texts_dir = f"data/{size}_{style}/train_bios.parquet"
    pretrain_texts = pd.read_parquet(pretrain_texts_dir)['content'].to_list()
    profiles_dir = f"data/{size}_{style}/people_profiles.parquet"
    profiles = pd.read_parquet(profiles_dir)
    name2profile = profiles.set_index("Fullname").T.to_dict()
    names = profiles["Fullname"].tolist()
    with open(f"data/{size}_{style}/check_data.jsonl", 'w', encoding='utf8') as f:
        pass

    name_iter = iter(names)
    current_name = next(name_iter)
    for text in tqdm(pretrain_texts):
        if current_name not in text:
            current_name = next(name_iter)
            assert current_name in text
        profile = name2profile[current_name]
        for value in profile.values():
            if type(value) != str: # ignore the last "Augmenttimes"
                continue
            value_index = text.find(value)
            info = {"prefix": text[:value_index],"answer": value}
            with open(f"data/{size}_{style}/check_data.jsonl", 'a+', encoding='utf8') as f:
                f.write(json.dumps(info) + "\n")

def pack_tokens(bio, qa, root_path, tokenizer, max_len=512, epoch=0):
    if os.path.exists(root_path + f"bio_tokens_{epoch}.pt") and os.path.exists(root_path + f"qa_tokens_{epoch}.pt"):
        print(f"Epoch {epoch} tokens already exists.")
        return
    random.seed(epoch)

    bio_tokens_list = []
    for bio_text in bio:
        tokens = tokenizer(text=bio_text, 
                           padding=False, 
                           truncation=False, 
                           max_length=max_len, 
                           return_tensors='pt',
                           return_token_type_ids=False,
                           return_attention_mask=False)['input_ids']
        bio_tokens_list.append(tokens)
    bio_tokens_list = bio_tokens_list * 2
    random.shuffle(bio_tokens_list)
    bio_tokens = torch.cat(bio_tokens_list, dim=1)
    bio_tokens_num = bio_tokens.size(1) // 2
    bio_reserve_tokens_num = bio_tokens_num + max_len - bio_tokens_num % max_len
    bio_tokens = bio_tokens[:, :bio_reserve_tokens_num]
    bio_tokens = bio_tokens.reshape(-1, max_len) # [batch_num, max_len]
    torch.save(bio_tokens, root_path + f"bio_tokens_{epoch}.pt")

    qa_tokens_list = []
    for qa_text in qa:
        tokens = tokenizer(text=qa_text, 
                           padding=False, 
                           truncation=False, 
                           max_length=max_len, 
                           return_tensors='pt',
                           return_token_type_ids=False,
                           return_attention_mask=False)['input_ids']
        qa_tokens_list.append(tokens)
    qa_tokens = torch.cat(qa_tokens_list, dim=1)
    expected_qa_tokens_num = bio_tokens.size(0) * 9 * max_len
    qa_tokens_num_one_times = qa_tokens.size(1)
    copy_times = expected_qa_tokens_num // qa_tokens_num_one_times + 1
    qa_tokens_list = [qa_tokens] * copy_times
    random.shuffle(qa_tokens_list)
    qa_tokens = torch.cat(qa_tokens_list, dim=1)
    qa_tokens = qa_tokens[:, :expected_qa_tokens_num]
    qa_tokens = qa_tokens.reshape(-1, max_len) # [batch_num, max_len]
    torch.save(qa_tokens, root_path + f"qa_tokens_{epoch}.pt")
    print(f"Epoch {epoch} tokens are saved: {bio_tokens.size(0)} bios, {qa_tokens.size(0)} qas.")

def tokenize_for_epochs(epochs, train_biographies, train_qa, root_path):
    tokenizer = LlamaTokenizer.from_pretrained("tokenizer.model")
    pool = Pool(10)
    for epoch in range(epochs):
        pool.apply_async(pack_tokens, args=(train_biographies, train_qa, root_path, tokenizer, 512, epoch))
    pool.close()
    pool.join()

def prepare_pretrain_data(
        size:str = "tiny", # the size of the data
        is_balanced:bool = True, # whether the data is balanced
        ):
    assert size in size2num.keys(), f"size should be one of {size2num.keys()}"
    style = "balanced" if is_balanced else "augmented"
    num = size2num[size]
    load_all_templates()
    load_all_possible_features()
    root_path = f"data/{size}_{style}/"
    os.makedirs(root_path, exist_ok=True)
    print(f"start loading/generating data of {size}_{style}...")

    # generate profiles for the fake people
    profiles = generate_profiles(num, root_path, is_balanced) 
    # generate biographies for the profiles
    train_biographies = generate_biographies(num, root_path, profiles, is_balanced)
    # generate question-answers for the fake people
    qa_data, train_qa, test_qa = generate_qa_profile(num, root_path, profiles)
    # tokenize the data for epochs
    tokenize_for_epochs(100, train_biographies, train_qa, root_path)
    # tokenizer = LlamaTokenizer.from_pretrained("tokenizer.model")
    # pack_tokens(train_biographies, train_qa, root_path, 
    #             tokenizer, max_len=512, epoch=0)

    construct_check_data(size=size, is_balanced=is_balanced)

if __name__ == "__main__":
    # prepare_pretrain_data("tiny", is_balanced=False)
    prepare_pretrain_data("tiny", is_balanced=True)
    # prepare_pretrain_data("small", is_balanced=False)
    # prepare_pretrain_data("small", is_balanced=True)

