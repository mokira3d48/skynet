venv:
	python3 -m venv env

install:
	pip install -r requirements.txt

dev:
	pip install -e .

test:
	pytest tests

run:
	python3 -m skynet

pep8:
    # autopep8 --in-place --aggressive --aggressive --recursive .
    # autopep8 --in-place --aggressive --aggressive example.py

