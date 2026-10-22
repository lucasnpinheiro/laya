"""Guard de pre-commit e commit-msg (.githooks/).

Tier rápido: sem modelo, sem servidor. Cada teste cria um repositório git temporário,
copia os hooks, aponta core.hooksPath para eles e confere o bloqueio ou a liberação.
Os padrões sensíveis são montados em tempo de execução (concatenação) para que este
arquivo não dispare o próprio guard quando for commitado.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOKS = REPO_ROOT / ".githooks"
ALLOWED = "2411995+lucasnpinheiro@users.noreply.github.com"
HOOK_FILES = ("pre-commit", "commit-msg", "denylist.local.example")


class Repo:
    def __init__(self, path: Path, env: dict):
        self.path = path
        self.env = env

    def git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=self.path, env=self.env,
                              capture_output=True, text=True, check=True)

    def write(self, name: str, content: str) -> None:
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def stage(self, name: str, content: str) -> None:
        self.write(name, content)
        self.git("add", name)

    def commit(self, msg: str = "feat: alteracao", env: dict | None = None,
               no_verify: bool = False) -> subprocess.CompletedProcess:
        run_env = dict(self.env)
        run_env.update(env or {})
        args = ["git", "commit", "-q", "-m", msg]
        if no_verify:
            args.append("--no-verify")
        return subprocess.run(args, cwd=self.path, env=run_env,
                              capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    path = tmp_path / "repo"
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, env=env, check=True)
    r = Repo(path, env)
    r.git("config", "user.name", "Teste")
    r.git("config", "user.email", ALLOWED)
    hooks_dir = path / ".githooks"
    hooks_dir.mkdir()
    for name in HOOK_FILES:
        shutil.copy2(HOOKS / name, hooks_dir / name)
    for name in ("pre-commit", "commit-msg"):
        (hooks_dir / name).chmod(0o755)
    r.git("config", "core.hooksPath", str(hooks_dir))
    r.stage("README.md", "base\n")
    result = r.commit("chore: base", no_verify=True)
    assert result.returncode == 0, result.stderr
    return r


def assert_blocked(result: subprocess.CompletedProcess) -> None:
    assert result.returncode != 0, result.stdout + result.stderr
    assert "BLOQUEADO" in result.stderr, result.stderr


def assert_allowed(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, result.stdout + result.stderr


# Conteúdos que devem bloquear o commit (nome do arquivo, conteúdo, rótulo).
SECRET_CASES = [
    ("segredo hf", "config.txt", "token = hf_" + "a1B2c3D4e5F6g7H8i9J0k1L2\n"),
    ("segredo ghp", "config.txt", "ghp_" + "x" * 20 + "\n"),
    ("segredo github_pat", "config.txt", "github_pat_" + "x" * 20 + "\n"),
    ("segredo sk-ant", "config.txt", "sk-ant-" + "api03-xyz\n"),
    ("segredo sk-", "config.txt", "sk-" + "a" * 24 + "\n"),
    ("segredo AWS", "config.txt", "AKIA" + "IOSFODNN7EXAMPLE\n"),
    ("chave privada", "chave.txt", "-----BEG" + "IN RSA PRIVATE KEY-----\n"),
    ("senha em claro", "config.txt", "pass" + "word = hunter2\n"),
    ("caminho macOS", "notas.txt", "/Us" + "ers/lucas/projeto/arquivo.txt\n"),
    ("caminho linux", "notas.txt", "/ho" + "me/dev/x/y\n"),
    ("caminho tmp claude", "notas.txt", "/private/tmp/" + "claude-123/out\n"),
    ("link de sessão", "notas.txt", "https://claude" + ".ai/code/session_abc\n"),
    ("e-mail gmail", "contato.txt", "fulano" + "@gmail.com\n"),
    ("arquivo .pem", "certs/server.pem", "conteudo qualquer\n"),
]


@pytest.mark.parametrize("label,name,content", SECRET_CASES, ids=[c[0] for c in SECRET_CASES])
def test_pre_commit_bloqueia(repo: Repo, label: str, name: str, content: str):
    repo.stage(name, content)
    assert_blocked(repo.commit())


def test_pre_commit_bloqueia_termo_da_denylist_local(repo: Repo):
    repo.write(".githooks/denylist.local", "# comentario\n\ncliente-secreto-teste\n")
    repo.stage("doc.md", "reunião com CLIENTE-SECRETO-TESTE ontem\n")
    assert_blocked(repo.commit())


def test_pre_commit_bloqueia_e_mail_de_autor_errado(repo: Repo):
    repo.git("config", "user.email", "dev@exemplo.com")
    repo.stage("codigo.py", "print('ola')\n")
    assert_blocked(repo.commit())


def test_commit_msg_bloqueia_link_de_sessao(repo: Repo):
    repo.stage("codigo.py", "print('ola')\n")
    msg = "feat: x\n\nhttps://claude" + ".ai/code/session_1"
    assert_blocked(repo.commit(msg))


def test_permite_alteracao_limpa(repo: Repo):
    repo.stage("codigo.py", "print('ola')\n")
    assert_allowed(repo.commit())


def test_permite_placeholder_de_senha(repo: Repo):
    repo.stage("exemplo.env.md", "password: <SENHA_EXEMPLO>\n")
    assert_allowed(repo.commit())


def test_permite_home_do_container(repo: Repo):
    repo.stage("Dockerfile", "ENV HF_HOME=/ho" + "me/laya/.cache/huggingface\n")
    assert_allowed(repo.commit())


def test_bloqueia_home_de_outro_usuario(repo: Repo):
    repo.stage("notas.txt", "/ho" + "me/joao/x\n")
    assert_blocked(repo.commit())


@pytest.mark.parametrize("content", [
    "Padrões: `pass" + "word=` e `senha:`\n",
    "lista: senha:, pass" + "word=, token\n",
    "use pass" + "word= sem valor\n",
], ids=["backtick", "lista-de-padroes", "sem-valor"])
def test_permite_documentacao_de_padroes(repo: Repo, content: str):
    repo.stage("doc.md", content)
    assert_allowed(repo.commit())


@pytest.mark.parametrize("content", [
    "pass" + "word=hunter2abc\n",
    "senha: Minha" + "Senha123\n",
], ids=["igual-com-valor", "senha-com-valor"])
def test_bloqueia_senha_com_valor_real(repo: Repo, content: str):
    repo.stage("config.txt", content)
    assert_blocked(repo.commit())


def test_pre_commit_ignora_linhas_removidas(repo: Repo):
    repo.stage("config.txt", "token = hf_" + "a1B2c3D4e5F6g7H8i9J0k1L2\n")
    assert_allowed(repo.commit("chore: segredo antigo", no_verify=True))
    repo.stage("config.txt", "token = removido\n")
    assert_allowed(repo.commit())


def test_allowed_email_configuravel(repo: Repo):
    outro = "outro@exemplo.com"
    repo.git("config", "user.email", outro)
    repo.git("config", "laya.allowedEmail", outro)
    repo.stage("codigo.py", "print('ola')\n")
    assert_allowed(repo.commit())


def test_setup_script_existe_e_e_executavel():
    script = REPO_ROOT / "scripts" / "setup-hooks.sh"
    assert script.is_file()
    assert os.access(script, os.X_OK)


def test_denylist_example_so_tem_exemplos_genericos():
    text = (HOOKS / "denylist.local.example").read_text()
    padroes = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert padroes == []
    assert "denylist.local" in (REPO_ROOT / ".gitignore").read_text()
