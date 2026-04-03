# Prompt Engineering for Multi-Agent Systems

## Project objective

This university assignment project demonstrates **prompt engineering** techniques for a simple **sentiment classification** task on the **IMDB** movie review dataset. It shows how different prompting strategies—**zero-shot**, **few-shot** (with examples drawn from data), and **chain-of-thought**—affect measured quality using **micro F1-score** against held-out labels.

The program also includes a small **tokenizer** exercise with **tiktoken**, a **LangChain** chat demo via **Groq** (OpenAI-compatible HTTP API, `ChatOpenAI` with a custom `base_url`) answering “What is an AI Agent?”, and a clean split between an **examples** pool and **gold** evaluation set via `train_test_split`.

## Technologies used

| Area | Library / tool |
|------|----------------|
| LLM API | **Groq** (OpenAI-compatible) via **LangChain** (`langchain-openai`, `ChatOpenAI` + `base_url`) |
| Environment | **python-dotenv** (loads `GROQ_API_KEY` from `.env`) |
| Token counting | **tiktoken** |
| Data | **Hugging Face datasets** (`imdb`), **pandas** |
| Evaluation | **scikit-learn** (`f1_score` with `average="micro"`), `train_test_split` |
| Interface | **Streamlit** (`app.py`) |

## Prerequisites

- Python 3.10+ recommended
- A **Groq** API key and access to the configured model (default: `openai/gpt-oss-120b` in `main.py`; change `CHAT_MODEL` if Groq renames models)

## How to run the project

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   ```

   Activate it:

   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`

2. **Install dependencies** — either file works:

   ```bash
   pip install -r requirements.txt
   ```

   Or install the project (pulls the same stack from **`pyproject.toml`**):

   ```bash
   pip install -e .
   ```

3. **Configure the API key** — copy `.env.example` to `.env`, then put your real key only in `.env`:

   ```bash
   cp .env.example .env
   ```

   ```env
   GROQ_API_KEY=gsk_...your_real_key...
   ```

   Create a key in the [Groq Console](https://console.groq.com/). The file **`.env` is listed in `.gitignore`** so it is **not** pushed to GitHub. Only **`.env.example`** (placeholder) belongs in the repository.

   **Publier sur GitHub sans exposer la clé**

   - Ne commitez jamais `.env` ni de clé dans le code ou le README.
   - Avant le premier commit : `git status` et vérifier que `.env` n’apparaît **pas** dans les fichiers suivis.
   - Si vous aviez déjà commis `.env` par erreur : supprimez-la dans la console Groq, créez une nouvelle clé, puis retirez l’historique (ou considérez le dépôt comme compromis).

   Do **not** commit real keys to git, paste them into source code, or share them in chats—**revoke any key that was exposed** and generate a new one.

   **Quick API test (bash)** — after `export GROQ_API_KEY=gsk_...`:

   ```bash
   curl https://api.groq.com/openai/v1/chat/completions -s \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $GROQ_API_KEY" \
     -d '{"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "Hello"}]}'
   ```

   **PowerShell** (set `$env:GROQ_API_KEY` first):

   ```powershell
   $h = @{ Authorization = "Bearer $env:GROQ_API_KEY"; "Content-Type" = "application/json" }
   $b = '{"model":"openai/gpt-oss-120b","messages":[{"role":"user","content":"Hello"}]}'
   Invoke-RestMethod -Uri "https://api.groq.com/openai/v1/chat/completions" -Method Post -Headers $h -Body $b
   ```

4. **Run** (CLI) :

   ```bash
   python main.py
   ```

5. **Interface Streamlit** (recommandé pour la démo) :

   ```bash
   streamlit run app.py
   ```

   Puis ouvrez l’URL affichée dans le terminal (souvent `http://localhost:8501`).

On the first run, the IMDB dataset may be **downloaded** by the `datasets` library (requires network access).

### Cost and runtime note

Evaluation calls the API once per review in the evaluation subset. By default, `MAX_EVAL_SAMPLES` in `main.py` limits how many gold rows are scored (to keep runs affordable for coursework). Set it to `None` in `main.py` to evaluate the full gold split (many API calls).

## Example results

Output is **non-deterministic** in general (even at low temperature), and scores depend on the model, prompts, and sample size. A typical run prints:

1. **Token count** for the fixed system prompt (tiktoken).
2. A short answer to **“What is an AI Agent?”** from `ChatOpenAI` (Groq backend).
3. Dataset sizes for the **examples** pool and **gold** split.
4. Three lines reporting **micro F1** for:

   - **Zero-shot**
   - **Few-shot**
   - **CoT** (chain-of-thought)

Example shape of the evaluation section (numbers are illustrative only):

```text
--- Evaluation (micro F1 vs gold labels) ---
Zero-shot — micro F1: 0.8200
Few-shot — micro F1: 0.8400
CoT — micro F1: 0.8300
```

Your numbers will differ based on model behavior, API version, and `MAX_EVAL_SAMPLES`.

## Project layout

| File | Role |
|------|------|
| `main.py` | Full pipeline: tokenizer, LLM demo, data load, prompts, evaluation |
| `app.py` | Interface **Streamlit** (tokenizer, chat Groq, aperçu IMDB, F1, classification unitaire) |
| `requirements.txt` | Minimal dependency list for this repo |
| `pyproject.toml` | **prompt-engineering-tp** metadata and dependencies (`pip install -e .`) |
| `.env.example` | Modèle sans secret — à copier vers `.env` en local |
| `.env` | **Local uniquement** (ignoré par Git) : votre vraie `GROQ_API_KEY` |
| `.gitignore` | Exclut `.env`, venv, caches, etc., pour éviter les fuites sur GitHub |
| `README.md` | This documentation |

## Academic use

Keep the code easy to read: adjust prompts, `FEW_SHOT_N`, `MAX_EVAL_SAMPLES`, or `test_size` in `split_examples_and_gold` to experiment and document findings in your report.
