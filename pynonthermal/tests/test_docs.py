"""Run the examples of the README and the notebooks, so that the text and the code agree.

The README test runs each fenced python block that follows the HTML comment "<!-- test -->". A block
that follows "<!-- test: continue -->" runs in the namespace of the block before it, because the text
continues that example. GitHub and PyPI do not show the comments. The fragments of the usage guide have
no comment, because each one needs the solver of the fragments before it.
"""

import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import pynonthermal

REPOSITORY_DIR = Path(__file__).parents[2]

README_BLOCK = re.compile(r"<!-- test(?P<continue>: continue)? -->\n```python\n(?P<source>.*?)```", re.DOTALL)

NOTEBOOKS = ("quickstart.ipynb", "fe_ionbalance_sn1a.ipynb")


def get_readme_examples() -> list[list[str]]:
    # each example is the list of its blocks, in the order of the README
    examples: list[list[str]] = []
    for match in README_BLOCK.finditer((REPOSITORY_DIR / "README.md").read_text(encoding="utf-8")):
        if match["continue"]:
            assert examples, "the first test block of the README continues no example"
            examples[-1].append(match["source"])
        else:
            examples.append([match["source"]])
    return examples


def run_sources(sources: list[str], filename: str) -> None:
    namespace: dict[str, object] = {"__name__": "__main__"}
    with warnings.catch_warnings():
        # the examples of elements above Ni use the Lotz formula, and the text says that it warns
        warnings.simplefilter("ignore", pynonthermal.LotzApproximationWarning)
        # plt.show() with the Agg backend of the tests (see conftest.py) shows nothing
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive", category=UserWarning)
        try:
            for source in sources:
                # the sources are the documentation of this repository
                exec(compile(source, filename, "exec"), namespace)  # noqa: S102
        finally:
            plt.close("all")


README_EXAMPLES = get_readme_examples()


def test_readme_has_test_blocks() -> None:
    # a regular expression that matches no block would let every README test pass with no test
    assert len(README_EXAMPLES) >= 5


@pytest.mark.parametrize("blocks", README_EXAMPLES, ids=[f"example{i}" for i in range(len(README_EXAMPLES))])
def test_readme_example(blocks: list[str]) -> None:
    run_sources(blocks, filename="README.md")


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_notebook(notebook: str) -> None:
    cells = json.loads((REPOSITORY_DIR / notebook).read_text(encoding="utf-8"))["cells"]
    # an IPython magic line is not Python, so the test leaves it out
    sources = [
        "".join(line for line in cell["source"] if not line.startswith("%"))
        for cell in cells
        if cell["cell_type"] == "code"
    ]
    run_sources(sources, filename=notebook)
