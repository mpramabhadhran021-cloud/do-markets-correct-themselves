import nbformat as nbf


def new_notebook():
    nb = nbf.v4.new_notebook()
    nb["cells"] = []
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


def md(nb, text):
    nb["cells"].append(nbf.v4.new_markdown_cell(text))


def code(nb, text):
    nb["cells"].append(nbf.v4.new_code_cell(text))


def save(nb, path):
    with open(path, "w") as f:
        nbf.write(nb, f)
