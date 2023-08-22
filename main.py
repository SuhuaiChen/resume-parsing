import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
import json
import re
import os
import io
import aiohttp
import asyncio
import time
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import uvicorn
from fastapi import Form, File, UploadFile, FastAPI
from typing import Annotated
from a2wsgi import ASGIMiddleware
from docx2python import docx2python


'''
pre-defined values
'''
app = FastAPI()
os.environ['TESSDATA_PREFIX'] = os.path.join(os.path.dirname(__file__), 'tess_data')
GPT4Engine = ''
GPT35Engine = ''
openai.api_type = ""
openai.api_version = ""
openai.api_base = ""
openai.api_key = ""
GPT35URL = ''

"""
General Helper functions
"""


# backoff
@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError,
                                   openai.error.RateLimitError, openai.error.ServiceUnavailableError,
                                   openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def auto_complete_json_brackets(s):
    if not s:
        return ''
    if s.strip()[-1] in ('}', ']'):
        return s

    index_of_last_comma = s.rfind(',')
    if index_of_last_comma != -1:
        s = s[:index_of_last_comma]

    cur_idx = len(s) - 1
    quote_num = 0
    while cur_idx > 0:
        if s[cur_idx] in ('}', ']', '{', '['):
            break
        if s[cur_idx] == "\"":
            quote_num += 1
        cur_idx -= 1
    if quote_num % 2 == 1:
        s += "\""

    stack = []
    for c in s:
        if c in ('{', '['):
            stack.append(c)
        if c in ('}', ']') and stack:
            stack.pop()

    while stack:
        left = stack.pop()
        if left == '{':
            s += '}'
        else:
            s += ']'

    return s


"""
Two ways to extract texts
1. extract_pdf, extract_docx (extract directly, fast, most of the time accurate)
2. ocr (slow, need to know the content language in advance, as accurate as method 1)
"""

# convert doc into pdf in batch: $ soffice --headless --convert-to pdf *.doc
# convert docx into pdf in batch $ soffice --headless --convert-to pdf *.docx


def extract(input_file):
    try:
        return extract_pdf(input_file)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")

    try:
        return ocr(input_file)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")

    try:
        return extract_docx(input_file)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")

    return ''


def extract_pdf(input_file):
    t0 = time.perf_counter()
    reader = PdfReader(io.BytesIO(input_file))
    pages = []
    for page in reader.pages:
        para = page.extract_text()
        para = re.sub(r'\n+', '\n', para)
        para = re.sub(r' +', ' ', para)
        # print(para)
        # print('*'* 80)
        pages.append(para)
    t1 = time.perf_counter()
    print('extraction time:', t1 - t0)
    with open('last_page.txt', 'w', encoding='utf-8') as f:
        f.write(pages[-1])
    return pages


def ocr(input_file):
    # pdf2jpeg
    pages = convert_from_bytes(input_file, fmt='jpeg')

    t0 = time.perf_counter()
    # jpg2txt
    page_list = []
    for page in pages:
        page_text = pytesseract.image_to_string(page, lang='eng+chi_sim+kor')
        page_list.append(page_text)
    t1 = time.perf_counter()
    print('jpg2text time:', t1 - t0)
    return page_list


def extract_docx(input_file):
    with docx2python(input_file) as docx_content:
        return re.sub(r'\n+', '\n', docx_content.text)

# ----------------------------------------------------------------------------------------------------------------------


"""
The GPT4 way to parse a resume: feed the whole text once. Currently unused due to latency
"""


def parse_one_pass(input_text):
    system_message = "You are an assistant designed to extract information. You always give detailed feedback." \
                     "Users will paste in a string " \
                     "and you will return a JSON file. The format of the JSON file should be " \
                     "[{\"name\": string, " \
                     "\"gender\": string, \"birth year\": string, \"phone number\": string, \"email\": string, " \
                     "\"desired salary:\": string, \"industry\": string, \"nationality\": string, " \
                     "\"current country:\": string, \"current city:\": string}," \
                     "{\"company name\": string, " \
                     "\"position\": string, \"duration\": string, \"achievement\": string, \"responsibility\": string}," \
                     "{\"school name\": string, " \
                     "\"education_level\": string, \"major\": string, \"duration\": string}," \
                     "{\"language spoken\": string, \"proficiency\": string}]"

    total_message = system_message + '\n\n' + input_text
    with open('prompt.txt', 'w', encoding='utf-8') as f:
        f.write(total_message)

    response = chat_completion_with_backoff(
        engine=GPT4Engine,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": "\n\"\"\"\n" + input_text}
        ],
        temperature=1,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=1000,
        stop=None
    )

    chat_ans = response['choices'][0]['message']['content']
    print(chat_ans)

    # find the first occurrence of [
    start_index = chat_ans.find('[')
    chat_ans = chat_ans[start_index:]

    # print(chat_ans)
    print(response['usage'])
    print('total cost:',
          response['usage']['completion_tokens'] * 0.06 / 1000 + response['usage']['prompt_tokens'] * 0.03 / 1000)

    try:
        data = json.loads(chat_ans)

        # for element in data:
        #     for key, value in element.items():
        #         print("{}: {}".format(key, value))
        return chat_ans

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return ''


def parse(input_file):
    try:
        pages = extract(input_file)
        resume_text = '\n\n'.join(pages)
        json_string = parse_one_pass(resume_text)
        return json_string

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return ''


"""
The GPT 3.5 way to parse a resume. Parse each page, and then merge. Different from the api call before, it is using
http request. 
"""


async def post_separate(pages):
    url = GPT35URL

    system_message = "You are an assistant designed to extract information." \
                     "Users will paste in a string " \
                     "and you will return a JSON file. The format of the JSON file should be " \
                     "{\"personal information\": " \
                     "{\"name\":string, " \
                     "\"gender\":string, \"birth year\":string, \"phone number\":string, \"email\":string, " \
                     "\"desired salary:\":string, \"industry\":string, \"nationality\":string, " \
                     "\"current country:\":string, \"current city:\":string}," \
                     "\"experience\": " \
                     "{\"company name\":string, " \
                     "\"position\":string, \"duration\":string, \"achievement\":string, \"responsibility\":string}," \
                     "\"education\": " \
                     "{\"school name\":string, " \
                     "\"education level\":string, \"major\":string, \"duration\":string}}," \
                     "There can be more than one education and experience sections. "

    async with aiohttp.ClientSession() as session:
        async def post(page_text):
            headers = {"Content-Type": "application/json", "api-key": openai.api_key,
                       "Authorization": "Bearer " + openai.api_key}
            messages = [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': page_text}]
            data = {
                'messages': messages,
                'top_p': 0.5,
                'max_tokens': 1500,
            }
            async with session.post(url=url, headers=headers, json=data) as response:
                return await response.json()

        return await asyncio.gather(*[
            post(page) for page in pages
        ])


def parse_separate(input_file):
    try:
        # page_list = ocr(input_file)
        page_list = extract(input_file)
        json_list = post_separate(page_list)
        return json_list

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return ''


"""
All the helper functions to safely post-process the json outputs of 'parse_separate'
"""


def is_not_empty_field(section_dict, field_name):
    return ((field_name in section_dict.keys()) and section_dict[field_name] and section_dict[field_name] != 'N/A'
            and section_dict[field_name] != 'Unknown')


def detect_splitter(text):
    patterns = [r'\.', r'\d\)', r'\n', r'-', r'。']
    arg_max = 0
    count_max = 0
    for idx, pattern in enumerate(patterns):
        cur_count = len(re.findall(pattern=pattern, string=text))
        if cur_count >= count_max:
            arg_max = idx
            count_max = cur_count
    return patterns[arg_max]


def to_list(field):
    if type(field) == list:
        return
    elif type(field) == str:
        field = re.split(pattern=detect_splitter(field), string=field)
        field = [i for i in field if i]
        return field
    else:
        return []


month_dict_eng = {
    'jan': '1', 'feb': '2', 'mar': '3', 'apr': '4', 'may': '5', 'jun': '6',
    'jul': '7', 'aug': '8', 'sep': '9', 'oct': '10', 'nov': '11', 'dec': '12'
}


def convert_date(s):
    pattern_year = r'\d\d\d\d'
    years = re.findall(pattern_year, s)

    s = s.lower()
    for m in month_dict_eng.keys():
        if m in s:
            s = re.sub(m, month_dict_eng[m], s)
    pattern_month = r'(?:^|\D)(\d\d?)(?:$|\D)'
    months = re.findall(pattern_month, s)

    start_year = years[0] if len(years) > 0 else ''
    start_month = months[0] if len(months) > 0 else ''
    end_year = years[1] if len(years) > 1 else ''
    end_month = months[1] if len(months) > 1 else ''

    print('formatted duration:', start_year, start_month, '-', end_year, end_month)
    # return start_year, start_month, end_year, end_month
    return start_year + ' ' + start_month + ' - ' + end_year + ' ' + end_month


def process_experience(experience_list):
    processed = []
    for experience in experience_list:
        if not experience or type(experience) is not dict:
            continue
        has_company_name = is_not_empty_field(experience, 'company name')
        if not has_company_name:
            continue
        has_position = is_not_empty_field(experience, 'position')
        has_duration = is_not_empty_field(experience, 'duration')
        has_achievement = is_not_empty_field(experience, 'achievement')
        has_responsibility = is_not_empty_field(experience, 'responsibility')
        processed.append({
            'company name': experience['company name'],
            'position': experience['position'] if has_position else '',
            'duration': convert_date(experience['duration']) if has_duration else '',
            'achievement': to_list(experience['achievement']) if has_achievement else [],
            'responsibility': to_list(experience['responsibility']) if has_responsibility else [],
            'achievement_raw': experience['achievement'] if has_achievement else '',
            'responsibility_raw': experience['responsibility'] if has_responsibility else ''
        })
    return processed


def process_education(education_list):
    processed = []
    for education in education_list:
        if not education or type(education) is not dict:
            continue
        has_school_name = is_not_empty_field(education, 'school name')
        if not has_school_name:
            continue
        has_education_level = is_not_empty_field(education, 'education level')
        has_major = is_not_empty_field(education, 'major')
        has_duration = is_not_empty_field(education, 'duration')
        processed.append({
            'school name': education['school name'],
            'education level': education['education level'] if has_education_level else '',
            'major': education['major'] if has_major else '',
            'duration': convert_date(education['duration']) if has_duration else '',
        })
    return processed


def get_sections_and_merge(json_list):
    personal_information_all = {
        "name": "",
        "gender": "",
        "birth year": "",
        "phone number": "",
        "email": "",
        "desired salary:": "",
        "industry": "",
        "nationality": "",
        "current country": "",
        "current city": ""
    }
    employment_experience_all = []
    education_all = []
    for json_file in json_list:
        print(json_file['choices'][0]['message']['content'])
        try:
            json_string = json_file['choices'][0]['message']['content']
            data = json.loads(auto_complete_json_brackets(json_string))
        except Exception as e:
            print('unable to load response,', e)
            continue

        try:
            if 'personal information' in data.keys():
                personal_information = data['personal information']
                for key, value in personal_information.items():
                    if key in personal_information_all.keys() and personal_information_all[key] == "" and value \
                            and value != 'N/A' and value != 'Unknown':
                        personal_information_all[key] = value
        except Exception as e:
            print('unable to merge personal information,', e)

        try:
            if 'experience' in data.keys():
                experience = data['experience']
                if type(experience) == list:
                    employment_experience_all.extend(experience)
                else:
                    employment_experience_all.append(experience)
        except Exception as e:
            print("unable to add experience,", e)

        try:
            if 'education' in data.keys():
                education = data['education']
                if type(education) == list:
                    education_all.extend(education)
                else:
                    education_all.append(education)
        except Exception as e:
            print("unable to merge education,", e)

    print('personal information: ', personal_information_all, '\n')
    print('employment experience: ', employment_experience_all, '\n')
    print('education: ', education_all, '\n')
    return personal_information_all, employment_experience_all, education_all


"""
the structured data is converted to a shortened CV. We query the shortened CV to get a summary
"""


def find_best_achievement(input_text):
    # system_message = "You are an HR assistant. Users will paste in a candidate profile, " \
    #                  "and you will answer the question: " \
    #                  "What makes the candidate unique?" \
    #                  "\nSupport your answers with figures if possible and return them in bullet points"

    system_message = "You are an HR assistant. Summarize 4 points of highlights from the resume below " \
                     "with the following categories:" \
                     "\n1. Summation of Experience" \
                     "\n2. Any promotions" \
                     "\n3. Any awards and achievements with facts" \
                     "\n4. Benefit statements" \
                     "Each highlight should be fewer than 150 characters."
    # "if you think the information is not enough, write \"lack information\" " \

    response = chat_completion_with_backoff(
        engine=GPT35Engine,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": "\n\"\"\"\n" + input_text}
        ],
        temperature=1,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=800,
        stop=None
    )

    chat_ans = response['choices'][0]['message']['content']
    print(chat_ans)
    print('\n\n', 'length of output string: ', len(chat_ans))
    return chat_ans


"""
Query the last page of a resume and see if there is a referee
"""


def get_reference(input_text):
    system_message = "You are an HR assistant designed to extract reference contact information of a candidate" \
                     "Users will paste in a string. And you will try to extract referee and contact"

    response = chat_completion_with_backoff(
        engine=GPT35Engine,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": "\n\"\"\"\n" + input_text}
        ],
        temperature=1,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=800,
        stop=None
    )
    chat_ans = response['choices'][0]['message']['content']
    return chat_ans[:150]


# -----------------------------------------------------------------------------------------------------------------------


"""
API Functions
"""


# root
@app.get('/')
def read_root():
    return {
        'text': 'hello world'
    }


# api call: resume file --> structured json
@app.post('/parse')
async def read_parse(file: UploadFile):
    f = await file.read()

    json_list = await parse_separate(f)

    personal_information_all, employment_experience_all, education_all = get_sections_and_merge(json_list)
    employment_experience_all = process_experience(employment_experience_all)
    education_all = process_education(education_all)

    personal_text = ''
    if personal_information_all['birth year']:
        personal_text += 'Born in ' + personal_information_all['birth year'] + '.'
    if personal_information_all['industry']:
        personal_text += 'Work in ' + personal_information_all['industry'] + '.'
    if personal_information_all['current country'] or personal_information_all['current city']:
        personal_text += 'Live in ' + personal_information_all['current city'] + ', ' + personal_information_all[
            'current country']

    employment_text = ''
    "Worked as a ... in ...; achievement: ... ; responsibility: ... duration: "
    for employment in employment_experience_all:
        work_text = '\n\nWorked'
        if employment['position']:
            work_text += ' as ' + employment['position']
        if employment['company name']:
            work_text += ' at ' + employment['company name'] + '.'
        if employment['achievement']:
            work_text += '\nAchievement: ' + employment['achievement_raw']
        if employment['responsibility']:
            work_text += '\nResponsibility: ' + employment['responsibility_raw']
        if employment['duration']:
            work_text += '\nDuration: ' + employment['duration']
        employment_text += work_text
        del employment['achievement_raw']
        del employment['responsibility_raw']

    education_text = ''
    "Studied..., .., at..., duration..."
    for i, education in enumerate(education_all):
        edu_text = "\n\nStudied"
        if education['major']:
            edu_text += ' ' + education['major']
        if education['education level']:
            edu_text += ' ' + education['education level']
        if education['school name']:
            edu_text += ' at ' + education['school name'] + '.'
        if education['duration']:
            edu_text += '\n' + education['duration']
        education_text += edu_text

    shortened_cv_text = personal_text + employment_text + education_text
    print('shortened:\n')
    print(shortened_cv_text)
    with open("temp.txt", 'w', encoding='utf-8') as f:
        f.write(shortened_cv_text)

    return {
        'personal information': personal_information_all,
        'employment experience': employment_experience_all,
        'education': education_all,
    }


@app.get('/summarize')
async def summarize():
    with open("temp.txt", encoding='utf-8') as f:
        shortened_cv = f.read()
    summary = find_best_achievement(shortened_cv)
    pattern = r'\d\.(.*)'
    points = re.findall(pattern=pattern, string=summary)
    print(points)
    summation_of_experience = ''
    any_promotions = ''
    any_awards_and_achievements_with_facts = ''
    benefit_statements = ''

    try:
        summation_of_experience = points[0]
    except Exception as e:
        print(e)

    try:
        any_promotions = points[1]
    except Exception as e:
        print(e)

    try:
        any_awards_and_achievements_with_facts = points[2]
    except Exception as e:
        print(e)

    try:
        benefit_statements = points[3]
    except Exception as e:
        print(e)
    return {
        'summation_of_experience': summation_of_experience[:150],
        'any_promotions': any_promotions[:150],
        'any_awards_and_achievements_with_facts': any_awards_and_achievements_with_facts[:150],
        'benefit_statements': benefit_statements[:150]
    }


@app.get('/reference')
async def reference():
    with open('last_page.txt') as f:
        last_page = f.read()
        print(last_page)
    reference_text = get_reference(last_page)
    return {
        "reference": reference_text
    }


@app.post('/parse_all')
async def parse_all(file: UploadFile):
    structured = await read_parse(file)
    summary = await summarize()
    reference_info = await reference()
    return {
        'structured data': structured,
        'summary': summary,
        'reference_info': reference_info
    }


@app.post('/cpr')
async def get_cpr(file: Annotated[UploadFile, File()], client_info: Annotated[str, Form()]='None',
                  kpi: Annotated[str, Form()]='None', education: Annotated[str, Form()]='None',
                  skills: Annotated[str, Form()]='None', target_company: Annotated[str, Form()]='None',
                  industry_insider_advice: Annotated[str, Form()]='None'):
    f = await file.read()
    pages = extract(f)
    resume_text = '\n\n'.join(pages)

    system_message = "You are an excellent headhunter. You task is to recommend candidates to your clients."
    system_message += (f'\nclient info: {client_info}'
                       + '\nclient\'s requirements:'
                       + f'\nkpi: {kpi}'
                       + f'\neducation: {education}'
                       + f'\nskills: {skills}'
                       + f'\ntarget companies: {target_company}'
                       + f'\nindustry insider advice: {industry_insider_advice}'
                       )

    system_message += f"\n\n Candidate Profile: {resume_text}"
    user_message = "\n\nWrite a candidate report that tailors to the client's requirements. The report should contain:" \
                   "\n1. a Highlights section with bullet points and statistical facts" \
                   "\n2. an Our Recommendation section with one sentence that summarizes the candidate's qualifications."

    print(system_message)
    with open("prompt.txt", 'w') as f:
        f.write(system_message)

    response = chat_completion_with_backoff(
        engine=GPT4Engine,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": "\n\"\"\"\n" + user_message}
        ],
        temperature=1,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=1000,
        stop=None
    )

    chat_ans = response['choices'][0]['message']['content']
    print(chat_ans)

    return chat_ans

wsgi_app = ASGIMiddleware(app)


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
