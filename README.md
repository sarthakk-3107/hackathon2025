# Mentor Matcher using Resume + LLM (deepseek-ai/deepseek-llm-7b-chat)

## 🧠 Project Description

This project leverages the power of LLMs to help individuals find mentors tailored to their career goals.  
It accepts a user's resume as input and automatically extracts all relevant skills using NLP techniques.  
The system then engages the user in a short interactive conversation using the LLM.  
It asks key questions like their current role, existing skills, and whether they're looking to switch industries.  
These answers, combined with resume data, create a personalized profile for the user.  
The profile is then matched against a curated mentor database.  
Mentors are filtered based on domain expertise, experience level, and skill overlap.  
The goal is to ensure relevant, inspiring, and helpful mentor connections.  
The mentor dataset is stored and queried using a vector database for efficient similarity search.  
The project uses **deepseek-ai/deepseek-llm-7b-chat** for all natural language understanding and generation tasks.  
LangChain handles prompt structuring and conversation flow with the user.  
ChromaDB powers the semantic search for mentor matching.  
The system is designed to be scalable, privacy-aware, and easy to integrate into career platforms.  
Ideal for students, early-career professionals, and career switchers.  
This AI-based mentoring solution provides personalized guidance at scale.
