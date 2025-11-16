.PHONY: up deps ingest api ui test

up:
	docker-compose up -d

deps:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

generate-data:
	python data/generate_cars_data.py

ingest:
	python src/ingestion.py

api:
	python api.py

ui:
	streamlit run streamlit_app.py

test:
	python tests/test_queries.py