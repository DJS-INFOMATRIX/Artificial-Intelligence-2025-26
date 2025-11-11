@echo off
echo ================================================
echo    ONE PIECE CHATBOT - STREAMLIT
echo ================================================
echo.
echo Starting the chatbot...
echo.
echo Once started, open: http://localhost:8501
echo Press Ctrl+C to stop
echo ================================================
echo.

cd /d %~dp0
streamlit run app.py web

pause
