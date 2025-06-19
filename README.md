
[![Watch my video to understand how QueryBuddy works!](https://img.youtube.com/vi/TnIAl9ICmtk/0.jpg)](https://www.youtube.com/watch?v=TnIAl9ICmtk)

Watch my video to understand how QueryBuddy works!

View the deployed web application over here: https://ruxquerybuddy.streamlit.app/

This is a RAG (Retrieval-Augmented Generation) Bot that answers questions based on your uploaded document in pdf format.Pinecone is used for storing and retrieving embeddings of the PDF chunks based on similarity to the query and Cohere is used for generating natural language answers by combining the retrieved context from Pinecone with the user’s query. This application is deployed and hosted on Streamlit.

(Currently I'm on the starter plan of Pinecone).

## Setup and Installation

To set up QueryBuddy on your local machine, follow these steps:

1. Clone the Repository:
   ```
   git clone https://github.com/rux0422/QueryBuddy-RAG-Bot.git
   cd QueryBuddy-RAG-Bot
   ```

2. Set Up a Virtual Environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install Dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set Up API Keys:
   Create a `config.py` file in the project root directory with the following content:
   ```python
   PINECONE_API_KEY = "your_pinecone_api_key"
   COHERE_API_KEY = "your_cohere_api_key"
   ```
   Replace the placeholder values with your actual API keys.

5. Initialize Pinecone Index:
   Ensure you have created an index named "qa-bot-index" in your Pinecone account.

6. Run the Application:
   ```
   streamlit run app.py
   ```

The application should now be running and accessible at http://localhost:8501.
