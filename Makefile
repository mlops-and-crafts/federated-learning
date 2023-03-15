setup_venv:
	python3.9 -m venv venv; . venv/bin/activate; pip install --upgrade pip; pip install -r requirements.txt

delete_venv:
	rm -rf venv

reset_venv: delete_venv setup_venv
