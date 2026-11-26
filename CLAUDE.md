# CLAUDE.md

## Project

The Machine is an offline, CPU-only AI companion that learns all the time from one teacher, through a webcam and speech. It is written in Python. The heavy lifting runs in native code: OpenCV DNN and CTranslate2.

The project is **public**.

## Commands

```bash
uv venv && uv pip install -e ".[dev]"     # setup
.venv/bin/python -m pytest -q              # tests, no camera needed, must stay green
.venv/bin/python -m the_machine bench      # performance, re-run after touching hot paths
.venv/bin/python -m the_machine run        # live system
```

## Architecture rules

- **Core is language-agnostic.** No English, or any other language, belongs in `core/` or `memory/`. Intent patterns, phrases and stopwords live only in `language/packs/<code>.py`. The core sees only `Intent.kind` and phrase keys. A new phrase key must be added to every pack.
- **`Mind` stays I/O-free.** Camera, microphone, TTS and display belong in `app.py` and `perception/`. This keeps `Mind` testable with fakes (see `tests/test_core.py::FakeFaces`).
- **All tunable numbers go in `config.py`.** Don't scatter magic numbers through the code.
- **No network at runtime.** Only `perception/models.py` (the one-time download) and `bench.py` (the sample image) may touch the network.
- **Privacy.** Store embeddings, not images. Never commit `data/`, `*.db` or model weights.
- **Minimal hardware is a feature.** Core dependencies are only `numpy`, `opencv-python` and `psutil`. Anything heavier goes in an optional extra in `pyproject.toml` and needs a graceful fallback.
- **Measure, don't claim.** A performance statement in README or docs must come from `bench` output.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) and keep messages **as simple as possible**: one short lowercase subject line and no body, unless the change really needs one.

```
feat: add voice embeddings
fix: reset track votes on forget
refactor: split vision tracker
docs: update teaching table
test: cover sleep stranger discovery
perf: cache prototype thresholds
chore: bump opencv
```

- Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `build`, `ci`.
- Make one logical change per commit, and run the tests before committing.

## Public repo hygiene

- This repo contains **only the system**. Research material such as paper drafts, unpublished results, reviewer notes and literature notes stays in a separate private repository.
- `eval/results/` is gitignored. Don't commit benchmark or evaluation outputs that will be reported in the paper.
- Mention research only in the short "Research" section of README.md, and don't add the paper's title, claims or numbers before publication.
