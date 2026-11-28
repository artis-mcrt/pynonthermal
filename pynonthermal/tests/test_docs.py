"""Run the examples in the README and the notebooks.

The README test starts an example at each "<!-- test -->" comment.
The "<!-- test: continue -->" comment runs the next block in the same namespace.
The notebook test compares selected results with the saved output at the precision that the notebook shows.
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

NOTEBOOK_RESULT_LABELS = {
    "quickstart.ipynb": ("heating:", "ionisation:", "excitation:", "sum:"),
    # The first three labels are inputs of the solver. The other labels are results of the ionisation balance:
    # the free electron density, and the ion fraction and the rate coefficient of three stages. The energy sum
    # is one to rounding also with an error in these results, so the sum alone does not check them. The saved
    # values of these lines are not near a rounding limit of the last digit that the notebook shows.
    "fe_ionbalance_sn1a.ipynb": (
        "iron number density:",
        "cobalt number density:",
        "deposition rate density:",
        "free electron density:",
        "  Fe II ",
        "  Fe III ",
        "  Fe IV ",
    ),
}


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
    assert len(README_EXAMPLES) >= 8


@pytest.mark.parametrize("blocks", README_EXAMPLES, ids=[f"example{i}" for i in range(len(README_EXAMPLES))])
def test_readme_example(blocks: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_sources(blocks, filename="README.md")


@pytest.mark.parametrize("notebook", NOTEBOOK_RESULT_LABELS)
def test_notebook(notebook: str, capsys: pytest.CaptureFixture[str]) -> None:
    cells = json.loads((REPOSITORY_DIR / notebook).read_text(encoding="utf-8"))["cells"]
    # an IPython magic line is not Python, so the test leaves it out
    sources = [
        "".join(line for line in cell["source"] if not line.startswith("%"))
        for cell in cells
        if cell["cell_type"] == "code"
    ]
    run_sources(sources, filename=notebook)
    calculated_lines = capsys.readouterr().out.splitlines()
    saved_lines = "".join(
        "".join(output["text"])
        for cell in cells
        for output in cell.get("outputs", [])
        if output.get("name") == "stdout"
    ).splitlines()
    for label in NOTEBOOK_RESULT_LABELS[notebook]:
        saved = [line for line in saved_lines if line.startswith(label)]
        assert len(saved) == 1, f"{notebook} must show one result for {label}"
        assert [line for line in calculated_lines if line.startswith(label)] == saved
