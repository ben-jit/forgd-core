test:
	python3 -m pytest

format:
	find . -name '*.py' -exec black {} \;
	autoflake -ri .

uninstall:
	conda env remove -n forgd_core -y

install:
	conda env create -f environment.yml

setup:
	python3 setup.py develop
