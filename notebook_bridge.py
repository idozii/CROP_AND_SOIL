import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Any, List, Optional, cast


class NotebookLoadError(Exception):
    """Raised when prediction callables cannot be loaded from notebook."""


_ALLOWED_ASSIGN_TARGETS = {"BASE_DIR", "logger", "model_manager"}
_ALLOWED_EXPR_CALLS = {
    "warnings.filterwarnings",
    "logging.basicConfig",
    "pd.set_option",
}


def _is_allowed_expr(node: ast.Expr) -> bool:
    if not isinstance(node.value, ast.Call):
        return False

    func = node.value.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        full_name = f"{func.value.id}.{func.attr}"
        return full_name in _ALLOWED_EXPR_CALLS

    return False


def _is_allowed_assign(node: ast.Assign) -> bool:
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return False
    return node.targets[0].id in _ALLOWED_ASSIGN_TARGETS


def _is_allowed_ann_assign(node: ast.AnnAssign) -> bool:
    return isinstance(node.target, ast.Name) and node.target.id in _ALLOWED_ASSIGN_TARGETS


def _sanitize_cell_code(source: str) -> str:
    """Keep only definitions and safe setup statements needed for inference."""
    parsed = ast.parse(source)
    safe_nodes = []

    for node in parsed.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            safe_nodes.append(node)
        elif isinstance(node, ast.Assign) and _is_allowed_assign(node):
            safe_nodes.append(node)
        elif isinstance(node, ast.AnnAssign) and _is_allowed_ann_assign(node):
            safe_nodes.append(node)
        elif isinstance(node, ast.Expr) and _is_allowed_expr(node):
            safe_nodes.append(node)

    return ast.unparse(ast.Module(body=safe_nodes, type_ignores=[])) if safe_nodes else ""


def _load_notebook_cells(notebook_path: Path) -> List[Dict[str, Any]]:
    with notebook_path.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    cells = notebook.get("cells", [])
    if not cells:
        raise NotebookLoadError("Notebook has no cells")
    return cells


def load_prediction_functions(notebook_path: Optional[Path] = None) -> SimpleNamespace:
    """
    Load predict_crop and predict_fertilizer from main.ipynb.

    The loader executes only sanitized code cells up to the unit test section to
    avoid running notebook side effects during app startup.
    """
    if notebook_path is None:
        notebook_path = Path(__file__).resolve().parent / "main.ipynb"

    if not notebook_path.exists():
        raise NotebookLoadError(f"Notebook not found: {notebook_path}")

    cells = _load_notebook_cells(notebook_path)
    namespace: Dict[str, Any] = {"__name__": "notebook_runtime"}

    for cell in cells:
        if cell.get("cell_type") == "markdown":
            markdown_text = "\n".join(cell.get("source", []))
            if "## 8. Unit Tests" in markdown_text:
                break
            continue

        if cell.get("cell_type") != "code":
            continue

        source_lines = cell.get("source", [])
        source = "\n".join(source_lines) if isinstance(source_lines, list) else str(source_lines)
        sanitized = _sanitize_cell_code(source)
        if not sanitized.strip():
            continue

        exec(compile(sanitized, str(notebook_path), "exec"), namespace)

    predict_crop = namespace.get("predict_crop")
    predict_fertilizer = namespace.get("predict_fertilizer")

    if not callable(predict_crop) or not callable(predict_fertilizer):
        raise NotebookLoadError("Could not load prediction functions from notebook")

    return SimpleNamespace(
        predict_crop=cast_callable(predict_crop, "predict_crop"),
        predict_fertilizer=cast_callable(predict_fertilizer, "predict_fertilizer"),
    )


def cast_callable(func: Any, name: str) -> Callable[..., Dict[str, Any]]:
    if not callable(func):
        raise NotebookLoadError(f"{name} is not callable")
    return cast(Callable[..., Dict[str, Any]], func)
