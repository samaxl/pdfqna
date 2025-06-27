# pdfqna
A sleek and intuitive Streamlit web application that allows users to upload a PDF document and ask natural language questions about its content. Powered by a retrieval-augmented generation (RAG) pipeline using BM25 retrieval and a Llama-4 model (hosted via SambaNova API).

1. Install the Python file (pdfbot.py)

2. Create a virtual environment and install the required packages:
```
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

3. Create a free SambaNova account and get your API key from [here](https://cloud.sambanova.ai/dashboard)

4. Create a .env file with the following variables:
```
OPENAI_API_KEY = [ENTER YOUR OPENAI API KEY HERE]
````

5. Run the Streamlit app:
```
streamlit run pdfbot.py
```
