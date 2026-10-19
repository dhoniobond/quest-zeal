# Quest Zeal

**An AI-powered question generation and assessment platform. Tutors upload study material, and Quest Zeal turns it into subjective and multiple-choice tests that students can take and get scored on automatically.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)

## Features

- **Tutor dashboard:** create a course and test by uploading a plain-text study document, and manage existing tests.
- **Subjective questions:** NLTK part-of-speech chunking pulls key concepts from the material and turns them into open-ended prompts such as *"Explain in detail…"*, *"Define…"* and *"Write a short note on…"*.
- **Multiple-choice questions:** a fine-tuned **T5** model generates question–answer pairs, and a second T5 model trained on the RACE dataset generates distractors. **sense2vec** similarity provides fallback distractors.
- **Automatic scoring:** written answers are scored by TF-IDF cosine similarity against the source material. MCQs are scored on correct answers.
- **Student dashboard:** shows available tests, which ones are complete, and scores.
- **Accounts:** registration and login with Flask-Login, with passwords hashed using bcrypt.

## How it works

```
Tutor uploads .txt ──► Uploaded Material/ ──┬─► Subjective: NLTK POS chunking ─► question templates
                                            │
                                            └─► Objective: T5 question/answer generation
                                                             ├─ T5 distractor model (RACE)
                                                             └─ sense2vec fallback distractors
Student answers ──► TF-IDF cosine similarity / answer key ──► score saved to SQLite
```

## Tech stack

| Layer | Tools |
| --- | --- |
| Web app | Flask, Flask-Login, Flask-WTF, Flask-Bcrypt, Jinja2 |
| Data | SQLite via Flask-SQLAlchemy |
| NLP / ML | PyTorch Lightning, Hugging Face Transformers (T5-small), sense2vec, spaCy, NLTK, scikit-learn |

## Getting started

Python 3.9 or 3.10 is recommended, matching the pinned dependencies.

```bash
git clone https://github.com/dhoniobond/quest-zeal.git
cd quest-zeal
python -m venv env
source env/bin/activate          # Windows: env\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then set a real SECRET_KEY
```

**Model files.** MCQ generation needs these files in `ml_models/`. To keep them somewhere else, set `QUESTZEAL_MODEL_DIR`. The files are not committed because of their size.

| File | Description |
| --- | --- |
| `multitask-qg-ag.ckpt` | T5-small fine-tuned for question and answer generation |
| `race-distractors.ckpt` | T5-small fine-tuned on RACE for distractor generation |
| `s2v_old/` | sense2vec vectors, available from [explosion/sense2vec](https://github.com/explosion/sense2vec#pretrained-vectors) |

Subjective question generation and scoring work without them.

**Run the app:**

```bash
flask run
```

Then open http://127.0.0.1:5000. The SQLite database is created automatically on the first request.

**Roles:** register a user named `admin` to get the tutor dashboard. Every other account is a student. Sample study material is in `Uploaded Material/`.

## Project structure

```
├── app/
│   ├── __init__.py                 Flask app, database, login manager
│   ├── config.py                   settings (SECRET_KEY from .env)
│   ├── models.py                   Users, Tests, Students tables
│   ├── forms.py                    login / register forms
│   ├── views.py                    routes and dashboards
│   ├── QA_Gen_Model.py             subjective question generator (NLTK)
│   ├── Objective_QA_Gen_Model.py   MCQ generator (T5 + sense2vec)
│   ├── QA_Score.py                 TF-IDF answer scoring
│   ├── static/                     CSS, fonts, images
│   └── templates/                  Jinja2 pages
├── Uploaded Material/              tutor uploads (with sample texts)
├── samples/                        extra sample material
├── run.py
└── requirements.txt
```

## Contributors

Built by [@YPrahasith](https://github.com/YPrahasith) and [@fizashaikhh](https://github.com/fizashaikhh). Originally developed at [YPrahasith/Quest_Zeal](https://github.com/YPrahasith/Quest_Zeal).
