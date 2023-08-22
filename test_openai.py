import openai
import os
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)


@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError,
                                   openai.error.RateLimitError, openai.error.ServiceUnavailableError,
                                   openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


openai.organization = ""
os.environ["OPEN_AI_API_KEY"] = ""
openai.api_key = os.getenv("OPEN_AI_API_KEY")


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
    with open('prompt.txt', 'w',encoding='utf-8') as f:
        f.write(total_message)

    response = chat_completion_with_backoff(
        model="gpt-4",
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
    print('total cost:', response['usage']['completion_tokens']*0.06/1000 + response['usage']['prompt_tokens']*0.03/1000)

    try:
        data = json.loads(chat_ans)

        # for element in data:
        #     for key, value in element.items():
        #         print("{}: {}".format(key, value))
        return chat_ans

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return ''





