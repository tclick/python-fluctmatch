.PHONY: clean virtualenv test docker dist dist-upload

SHELL = /bin/zsh
BASENAME = python-fluctmatch
CONDA_DIR = ${HOME}/.conda/envs/${BASENAME}
clean:
	find . -name '*.py[co]' -delete

conda:
	conda create -y -p ${CONDA_DIR} pip
	conda activate $(basename ${CONDA_DIR}) && pip install -r requirements-dev.txt && python setup.py develop

virtualenv:
	virtualenv --prompt '|> python-fluctmatch <| ' env
	env/bin/pip install -r requirements-dev.txt
	env/bin/python setup.py develop
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
