import ast
import hashlib

from config import ALLOWED_MODULES  # Single source of truth

FORBIDDEN_FUNCTIONS = {"eval", "exec", "open", "subprocess", "os", "sys", "socket", "__import__", "compile", "globals", "locals"}

class ASTSafetyValidator(ast.NodeVisitor):
    def __init__(self):
        self.errors = []

    def visit_Import(self, node):
        for alias in node.names:
            base_mod = alias.name.split('.')[0]
            if base_mod not in ALLOWED_MODULES:
                self.errors.append(f"Forbidden import: '{alias.name}'. Only {ALLOWED_MODULES} allowed.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            base_mod = node.module.split('.')[0]
            if base_mod not in ALLOWED_MODULES:
                self.errors.append(f"Forbidden import from module: '{node.module}'.")
        self.generic_visit(node)

    def visit_Call(self, node):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in FORBIDDEN_FUNCTIONS:
            self.errors.append(f"Forbidden function call: '{func_name}'.")

        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("__"):
            self.errors.append(f"Dunder attribute access '{node.attr}' is prohibited.")
        self.generic_visit(node)


def validate_strategy_code(code_str: str) -> tuple[bool, str]:
    """
    Parses Python code into an AST and checks for safety violations.
    Returns (is_safe, error_message).
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Syntax Error in generated Python code: {e}"

    validator = ASTSafetyValidator()
    validator.visit(tree)

    if validator.errors:
        return False, "AST Safety Violations:\n - " + "\n - ".join(validator.errors)

    # Ensure required function signature is defined
    has_target_fn = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_signals":
            has_target_fn = True
            break

    if not has_target_fn:
        return False, "Generated code must define the entrypoint function 'generate_signals(df, params)'."

    return True, "AST Safety Check Passed."


def get_strategy_fingerprint(code_str: str) -> str:
    """
    Generates a deterministic SHA-256 fingerprint for deduplication (PRD Section 7.7).
    """
    # Strip whitespace and comments for normalization
    lines = [line.split('#')[0].strip() for line in code_str.splitlines() if line.strip()]
    normalized_code = "\n".join(lines)
    return hashlib.sha256(normalized_code.encode('utf-8')).hexdigest()[:16]
