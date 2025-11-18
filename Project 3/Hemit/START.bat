@echo off
echo ================================================
echo           GAMES RAG CHATBOT - STREAMLIT
echo ================================================
echo.
echo Starting the Games Knowledge Chatbot...
echo.
echo Once started, open: http://localhost:8501
echo Press Ctrl+C to stop
echo ================================================
echo.

cd /d %~dp0
streamlit run app.py -- web

pause

