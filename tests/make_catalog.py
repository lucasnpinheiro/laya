#!/usr/bin/env python3
"""Gera o catalogo de testes a partir de uma execucao real.

Le tests/reports/junit.xml (resultado + duracao) e os proprios arquivos de teste (a
primeira linha do docstring de cada teste), escreve tests/reports/catalog.md e injeta a
mesma tabela em tests/RESULTS.md, entre os marcadores CATALOG. Gerar em vez de transcrever
evita que a documentacao e a suite divirjam quando um teste e renomeado.

    ./tests/run.sh --run-slow --junitxml=tests/reports/junit.xml
    uv run --no-project python tests/make_catalog.py
"""

from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

TESTS_DIR = Path(__file__).parent
JUNIT = TESTS_DIR / "reports" / "junit.xml"
OUT = TESTS_DIR / "reports" / "catalog.md"
RESULTS = TESTS_DIR / "RESULTS.md"
START = "<!-- CATALOG:START -->"
END = "<!-- CATALOG:END -->"

FILE_TITLES = {
    "test_mcp_contract.py": "Contrato MCP",
    "test_decision_quality.py": "Qualidade de decisão",
    "test_autostart.py": "Auto-start do container",
    "test_spec_research.py": "Triagem de requisitos (spec-research)",
}


def summaries() -> dict[tuple[str, str], str]:
    """Primeira frase do docstring de cada funcao de teste, por (arquivo, nome)."""
    found = {}
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            doc = ast.get_docstring(node) or ""
            first = doc.strip().split("\n")[0].strip() if doc else ""
            found[(path.name, node.name)] = first
    return found


def results() -> list[dict]:
    rows = []
    for case in ET.parse(JUNIT).getroot().iter("testcase"):
        classname = case.get("classname", "")
        file_name = classname.split(".")[-1] + ".py" if classname else "?"
        outcome = "passou"
        for child in case:
            if child.tag == "failure" or child.tag == "error":
                outcome = "FALHOU"
            elif child.tag == "skipped":
                outcome = "pulado"
        rows.append({
            "file": file_name,
            "name": case.get("name", ""),
            "seconds": float(case.get("time", 0.0)),
            "outcome": outcome,
        })
    return rows


def main() -> None:
    docs = summaries()
    rows = results()

    lines = ["<!-- GERADO por tests/make_catalog.py - nao editar a mao -->", ""]
    total = len(rows)
    passed = sum(r["outcome"] == "passou" for r in rows)
    elapsed = sum(r["seconds"] for r in rows)
    lines.append(f"{passed} de {total} testes passaram — {elapsed:.1f}s somados.")
    lines.append("")

    for file_name in FILE_TITLES:
        group = [r for r in rows if r["file"] == file_name]
        if not group:
            continue
        seconds = sum(r["seconds"] for r in group)
        lines.append(f"### `{file_name}` — {FILE_TITLES[file_name]} ({len(group)} testes, {seconds:.1f}s)")
        lines.append("")
        lines.append("| Teste | Verifica | Resultado | Tempo |")
        lines.append("|---|---|---|---|")
        for row in group:
            summary = docs.get((file_name, row["name"]), "") or "—"
            lines.append(
                f"| `{row['name']}` | {summary} | {row['outcome']} | {row['seconds']:.2f}s |"
            )
        lines.append("")

    catalog = "\n".join(lines)
    OUT.write_text(catalog)
    print(f"{OUT} — {passed}/{total} passaram")

    if RESULTS.exists():
        doc = RESULTS.read_text()
        if START in doc and END in doc:
            head, rest = doc.split(START, 1)
            _, tail = rest.split(END, 1)
            RESULTS.write_text(f"{head}{START}\n{catalog}\n{END}{tail}")
            print(f"{RESULTS} — catálogo injetado")
        else:
            print(f"{RESULTS} — marcadores {START}/{END} ausentes, nada injetado")


if __name__ == "__main__":
    main()
