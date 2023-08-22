# testApp_V3

Setup 

(run the following commands in the app directory)
1. Install pip

```
python -m pip install --user --upgrade pip
python -m pip --version
```

2. Install dependencies

`python -m pip install -r requirements.txt`

3. Start server

`python -m main.py`


---

API Documentation

- Default IP: http://127.0.0.1:8000
 

There are 5 API Functions
 

- POST `parse`: pdf file -> structured data
  - request body: {file: Bytes}
  - When executed, it also stores "temp.txt" and "last_page.txt" locally. "temp.txt" is the shortened CV and "last_page.txt" is the last page of the cv file in text format
- GET `summarize`
  - It reads the local "temp.txt" file and outputs the highlights
- GET `reference`
  - It reads the local "last_page.txt" file and outputs the reference info
- POST `parse_all`
  - request body: same as parse
  - It calls parse, summarize, and reference consecutively and outputs a json that contains structured data, highlights, and reference info
- POST `cpr`
   - request body: {file: Bytes, client_info: String, client's requirements: String, kpi: String, education: String, 
  skills: String, target_company: String, industry_insider_advice: String}
   - resume + ppr info => cpr

---
Components

`main.py`: The app

`test_openai.py`: Directly calling Openai api instead of through Azure, currently gpt4 access not granted

`tess_data`: Data required by the ocr model

`classification`: dataset, script to create the dataset, and notebook to train the classification model
