.PHONY: clean virtualenv test docker dist dist-upload

SHELL = /bin/bash
BASENAME = python-fluctmatch
CONDA_DIR = ${HOME}/.conda/envs/${BASENAME}
clean:
	find . -name '*.py[co]' -delete

conda:
	conda create -y -c conda-forge -p ${CONDA_DIR} pip python=3.8
	source activate ${BASENAME} && pip install -r requirements.txt
	source activate ${BASENAME} && pip install .

conda-dev:
	conda create -y -c conda-forge -p ${CONDA_DIR} pip python=3.8
	source activate ${BASENAME} && pip install -r requirements.txt
	source activate ${BASENAME} && pip install -r requirements-dev.txt
	source activate ${BASENAME} && pip install -e .

virtualenv:
	virtualenv --prompt '|> python-fluctmatch <| ' env
	env/bin/pip install -r requirements.txt
	env/bin/pip install -e .
	@echo
	@echo "VirtualENV Setup Complete. Now run: source env/bin/activate"
	@echo

virtualenv-dev:
	virtualenv --prompt '|> python-fluctmatch <| ' env
	env/bin/pip install -r requirements.txt
	env/bin/pip install -r requirements-dev.txt
	env/bin/pip install -e .
	@echo
	@echo "VirtualENV Setup Complete. Now run: source env/bin/activate"
	@echo

test:
	python -m pytest \
		-v \
		--cov=${BASENAME} \
		--cov-report=term \
		--cov-report=html:coverage-report \
		tests/

docker: clean
	docker build -t ${BASENAME}:latest .

dist: clean
	rm -rf dist/*
	python setup.py sdist
	python setup.py bdist_wheel

dist-upload:
	twine upload dist/*
