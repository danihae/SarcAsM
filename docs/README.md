To compile the docs, you would need to install (in the environment with working SarcAsM)

- [Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html)
- The Sphinx extensions [nbsphinx](https://nbsphinx.readthedocs.io/en/0.2.15/installation.html) and [autoapi](https://sphinx-autoapi.readthedocs.io/en/latest/)
- The [IPython lexer](https://nbsphinx.readthedocs.io/en/0.2.15/installation.html#Pygments-Lexer-for-Syntax-Highlighting)
- [ReadTheDocs theme](https://pypi.org/project/sphinx-rtd-theme/)

and then run `make html` in this directory.
