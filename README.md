# SOCIAL MEDIA TEXT ANALYSIS: Investigating viewer emotion and sentiment dynamics under YouTube videos

## 🎯 Project Goal
- The objective of this project is to develop an interactive Dashboard application that allows for the automated analysis of user opinions under any YouTube video. This project combines Natural Language Processing (NLP) techniques with modern web development frameworks.

## 🚀 Core Features
- Real-time Sentiment Analysis: Automatically fetching data and classifying emotions into Positive, Negative, and Neutral categories as soon as a link is provided.
- Interactive Time-Series Dashboard: A visual representation of mood changes over time, allowing the detection of "crisis moments" or sudden spikes in viewer approval.
- Automated Topic Modeling: Utilizing algorithms to group comments into thematic clusters, such as "music," "acting," or "plot," to quickly identify what users are discussing most.
- Multilingual Support: Integrating models capable of analyzing comments in various languages (e.g., Polish and English), which serves as a significant research element exceeding basic lecture material.

## 🧠 Research Aspect (AI Experiments)
- Baseline (VADER): A dictionary-based classifier optimized for social media that recognizes emojis, punctuation, and slang.
- Advanced Model (Transformers): Utilizing pre-trained models from the Hugging Face library (e.g., XLM-RoBERTa) for deep semantic analysis.
- Experimental Fine-tuning: Attempting to adjust models for specific pop-culture contexts or using techniques like LoRA (Low-Rank Adaptation).

## 🛠️ Technical Stack
- Backend: Python, FastAPI
- Frontend: React + Vite for a highly interactive user interface
- AI/NLP: NLTK (VADER), Hugging Face Transformers, Pandas, scikit-learn
- Data Source: YouTube Data API v3
