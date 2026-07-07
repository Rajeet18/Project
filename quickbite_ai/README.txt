QuickBite AI - Intelligent Food Delivery Assistant

Step 1: Install packages
pip install -r requirements.txt

Step 2: Create .env file
Copy .env.example and rename it to .env
Add your OpenAI API key:
OPENAI_API_KEY=your_api_key_here

Step 3: Train ML model
python train_model.py

Step 4: Run Flask API
python flask_api.py

Step 5: Open another terminal and run Streamlit app
streamlit run app.py

Project includes:
- Streamlit sidebar for address and dietary preference
- Session memory using st.session_state
- Flask /predict delivery API
- Custom LangChain tool get_delivery_estimate
- JSON menu search with 10 items
- Few-shot prompt examples
- Chat history with Clear Chat button
