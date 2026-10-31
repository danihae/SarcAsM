"""The lazy public names of ``sarcasm`` stay visible to IDEs and type checkers."""
import ast
from pathlib import Path

import sarcasm


def test_type_checking_imports_match_lazy_exports():
    """Every lazy export is also imported, from the same module, under ``if TYPE_CHECKING:``."""
    tree = ast.parse(Path(sarcasm.__file__).read_text())
    block = next(node for node in tree.body
                 if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING")
    static = {alias.name: "." + imp.module
              for imp in block.body if isinstance(imp, ast.ImportFrom)
              for alias in imp.names}
    assert static == sarcasm._EXPORTS
