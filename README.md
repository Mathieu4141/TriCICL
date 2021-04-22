

## Install 

### Poetry

[pypoetry doc](https://python-poetry.org/) is very well written and detailed.

*First, be sure to not be in a virtual env.*

To [install poetry](https://python-poetry.org/docs/#installation) with the right version :
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_VERSION=1.1.6 python`
(on Windows, from PowerShell) `(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | POETRY_VERSION=1.1.6 python -
`

By default, poetry will create a virtualenv in a [cache-dir folder](https://python-poetry.org/docs/configuration/#cache-dir-string). To have it created in the repository, under a `.venv` folder, you need to first run `poetry config virtualenvs.in-project true` (https://python-poetry.org/docs/configuration/#virtualenvsin-project-boolean).
Then go to our repository, and run `poetry install`. It will create a virtualenv that can be used in PyCharm, with all the dependencies needed.

### MNIST download fix

There may be an issue when downloading MNIST. To fix it, run:

```bash
cd ~/.avalanche/data/mnist/
wget www.di.ens.fr/~lelarge/MNIST.tar.gz
tar -zxvf MNIST.tar.gz
```