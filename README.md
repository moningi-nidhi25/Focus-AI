<div align="center">

# FocusAI

### Study with intention. Browse with focus.

**An AI-powered, context-aware learning assistant that evaluates online content against your current study goals.**

<p>
  <img src="https://img.shields.io/badge/AI-Context--Aware-7C3AED?style=for-the-badge" alt="Context-aware AI">
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="React and TypeScript">
  <img src="https://img.shields.io/badge/Browser-Chrome%20Extension-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Chrome extension">
</p>

<p>
  <a href="#-the-idea">The Idea</a> •
  <a href="#-features">Features</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-privacy-and-access-control">Privacy</a> •
  <a href="#-testing--evaluation">Testing</a>
</p>

</div>

---

## 🎯 The Idea

When studying online, useful material and distractions often live on the same websites. A traditional blocklist cannot tell the difference between a lecture that supports your assignment and an unrelated video on the same platform.

**FocusAI takes your study context into account.** It evaluates the content you are viewing against your topic, assignment, or learning objectives and categorizes it as:

<table>
  <tr>
    <td align="center" width="33%">
      <h3>🟢 Relevant</h3>
      <p>Content that supports your current learning goal. Allowed.</p>
    </td>
    <td align="center" width="33%">
      <h3>🟡 Partially Relevant</h3>
      <p>Content that may be useful but is not fully aligned. Warned.</p>
    </td>
    <td align="center" width="33%">
      <h3>🔴 Distracting</h3>
      <p>Content that is unrelated to your current study context. Blocked, with a logged "continue anyway" option.</p>
    </td>
  </tr>
</table>

The goal is not to block entire websites. It is to make content decisions based on **what you are trying to learn right now**.

## ✨ Features

<table>
  <tr>
    <td width="50%">
      <h3>🎯 Study Context</h3>
      Define a study topic, learning objectives and optional course material (PDF, DOCX, TXT, MD) so content can be evaluated against your current goal.
    </td>
    <td width="50%">
      <h3>🧠 Semantic Relevance</h3>
      Uses text embeddings to estimate how closely page content matches your study context.
    </td>
  </tr>
  <tr>
    <td>
      <h3>🤖 Contextual AI Evaluation</h3>
      Uses Google Gemini selectively when the initial relevance signal is ambiguous or the available content needs additional reasoning.
    </td>
    <td>
      <h3>🧭 Intelligent Filtering</h3>
      Allows, warns or blocks individual pages, never whole domains. YouTube itself is not blocked; an unrelated video is.
    </td>
  </tr>
  <tr>
    <td>
      <h3>💡 Explainable Decisions</h3>
      Every decision comes with a plain-language explanation, a relevance percentage, matching concepts and a sentence-level relevance heatmap.
    </td>
    <td>
      <h3>🌐 Browser Integration</h3>
      A Chrome extension extracts page content (including YouTube titles, descriptions and transcripts), shows allow/warn/block overlays and can toggle the heatmap on the page.
    </td>
  </tr>
  <tr>
    <td>
      <h3>📊 Learning Analytics</h3>
      Tracks study time, distractions, blocked pages and tab switches, and turns them into a Focus Score and a Productivity Score with personalized insights.
    </td>
    <td>
      <h3>📉 Focus Decay Prediction</h3>
      An LSTM model forecasts focus over the next activity windows, draws a Focus Decay Curve with an uncertainty band and sends a browser notification before a predicted dip.
    </td>
  </tr>
  <tr>
    <td colspan="2">
      <h3>👥 Role-Based Dashboards</h3>
      Separate student, teacher and administrator views. Teachers see anonymized cohort-level analytics only; administrators see platform totals.
    </td>
  </tr>
</table>

## 🧠 How It Works

```text
┌──────────────────────────┐
│  1. Set a study context  │
│  Topic + learning goals  │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│  2. Capture page content │
│  Chrome extension        │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│  3. Analyze relevance    │
│  Text embeddings         │
│  + adaptive scoring      │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│  4. Resolve ambiguity    │
│  Gemini evaluation       │
│  when needed             │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│  5. Decide and explain   │
│  Allow / Warn / Block    │
│  + explanation + heatmap │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│  6. Learn from activity  │
│  Focus and Productivity  │
│  scores, LSTM forecast,  │
│  role-based dashboards   │
└──────────────────────────┘
```

### Relevance engine

FocusAI combines two kinds of signals:

1. **Semantic similarity:** the `sentence-transformers/all-MiniLM-L6-v2` model creates text embeddings used to compare the study context with page content. If the model is not installed, a built-in lexical fallback is used (works offline but is weaker on synonyms).
2. **Contextual reasoning:** Google Gemini is used selectively for ambiguous cases and for pages with very little text, rather than for every classification.

The signals are combined into a 0-100 relevance score: **75 or above is allowed, 30 to 75 triggers a warning, below 30 is blocked.** Pages with nothing readable are never blocked.

### Explainability

Explanations and heatmaps use **sentence-level semantic similarity**, which is valid for any embedding backend. Integrated Gradients and LRP are deliberately not claimed: they need gradient access to a model we control and are not valid for an external LLM API.

### Scores

The formulas are documented at the top of `backend/app/services/metrics_service.py`.

- **Focus Score (0-100)** = 100 x (0.50 x relevance ratio + 0.25 x distraction term + 0.25 x tab-switch term)
- **Productivity Score (0-100)** = 100 x (0.60 x share of time on relevant content + 0.40 x progress toward a 25-minute focus block)

### Focus decay model

A two-layer LSTM reads the last 10 activity snapshots (focus score, productivity score, dwell time, tab switches, distractions) and predicts focus for the next 5 windows. The uncertainty band comes from validation error. The model is currently **trained on synthetic study sessions**; see [Known limitations](#-known-limitations).

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite, Recharts |
| Backend API | Python 3.12+, FastAPI, Uvicorn, SQLAlchemy, managed with `uv` |
| Database | PostgreSQL on Supabase (SQLite is the local default) |
| Semantic analysis | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Contextual evaluation | Google Gemini API |
| Machine learning | PyTorch (LSTM focus-decay model), NumPy |
| Authentication | JWT login tokens, bcrypt password hashing |
| Browser integration | Chrome Extension (Manifest V3) |
| Development tools | Git, GitHub, VS Code, pytest |

## 🏗️ Project Structure

```text
FocusAI/
├── backend/
│   ├── app/
│   │   ├── ai/          # embeddings, LLM review, relevance engine, explainability
│   │   ├── api/         # routes: auth, sessions, relevance, activity, prediction, dashboards
│   │   ├── ml/          # LSTM model, synthetic data, trained model file
│   │   ├── models/      # database tables
│   │   ├── schemas/     # request / response shapes
│   │   ├── services/    # content extraction, scores, predictions, dashboard analytics
│   │   └── utils/       # database and security helpers
│   ├── scripts/         # create_user, seed_demo_dashboard, train_lstm, eval_relevance
│   ├── tests/           # pytest suite
│   └── main.py
├── frontend/            # React + TypeScript app and role dashboards
├── extension/           # Chrome extension
├── extension-tests/     # Extension-related tests
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- `uv` for the backend (it installs the right Python version itself)
- Node.js and npm for the frontend
- Google Chrome to load the browser extension
- A Google Gemini API key if contextual evaluation is enabled
- A Supabase project (optional; without it a local SQLite file is used)

### 1. Clone the repository

```bash
git clone https://github.com/pawariarushil/FocusAI.git
cd FocusAI
```

### 2. Start the backend

```bash
cd backend
uv sync --extra ml
```

`--extra ml` installs the semantic embedding model (about 2 GB). Leave it out to use the lighter lexical fallback.

Create `backend/.env` (it is ignored by Git; never commit it):

```text
APP_NAME=FocusAI
ENVIRONMENT=development
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/postgres
LLM_PROVIDER=gemini
LLM_API_KEY=your_gemini_key
JWT_SECRET=a_long_random_string
```

Generate the secret with:

```bash
uv run python -c "import secrets; print(secrets.token_urlsafe(48))"
```

If `DATABASE_URL` is left out, a local SQLite file is used. Tables are created automatically on startup. Then start the API:

```bash
uv run uvicorn main:app --reload
```

Interactive API documentation is at `http://127.0.0.1:8000/docs`.

### 3. Start the frontend

Open a second terminal from the project root:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### 4. Load the Chrome extension

1. Open Chrome and visit `chrome://extensions`.
2. Enable **Developer mode**.
3. Select **Load unpacked** and choose the project's `extension/` directory.
4. Log in to the web app, then **refresh the web app page once** so the extension receives your login token.
5. Start a study session in the web app. Open new tabs (or refresh old ones), because Chrome only attaches extensions to pages loaded after they were installed or reloaded.

If you edit extension files, click the reload arrow on the extension card, then refresh the web app and your tabs.

### 5. Accounts and roles

- Students and teachers sign up in the web app. Administrators cannot sign up there.
- Create an administrator, or set a password on an existing account:

```bash
  cd backend
  uv run python scripts/create_user.py
```

- Optional demo data (a teacher, six students and synthetic activity, all with `@demo.focusai.dev` emails):

```bash
  uv run python scripts/seed_demo_dashboard.py
```

  Delete these demo accounts before any public deployment.

### 6. Train the focus-decay model

```bash
cd backend
uv run python scripts/train_lstm.py
```

This writes `app/ml/artifacts/focus_lstm.pt` (a trained copy is already included in the repository).

## 🔒 Privacy and Access Control

- Every API endpoint requires a login. Roles are read from the database on every request, never trusted from the token.
- Students can only read and change their own sessions. Another student's session returns "not found".
- Teachers see aggregated cohort statistics only: no names, emails or browsing history. Cohorts with fewer than 3 students are hidden.
- Administrators see platform totals only.
- Activity tracking sends hostnames and counts, never full URLs or page text. Page text is sent only to the relevance engine for analysis.
- Keep `.env` files out of version control and never put API keys in frontend or extension code.

## 🧪 Testing & Evaluation

```bash
cd backend
uv run pytest -q                           # 142 tests: API, relevance engine, security / RBAC, scores, LSTM
uv run python scripts/eval_relevance.py    # precision / recall / F1 on the labeled set

cd ../frontend
npm run build
npm run lint
```

### Word-level explanations (Integrated Gradients)

Besides the sentence heatmap, the extension can explain *which words* of a passage move its embedding toward or away from
your study goal, using genuine Integrated Gradients on the pretrained all-MiniLM-L6-v2 model. It explains only the
embedding-cosine component (not the keyword term or the final decision), shows raw signed attributions with a
display-only colour scale, and is fully optional. Details, limits and the API: `backend/README.md`. It needs the
`ml` extra (`uv sync --extra ml`); with the lexical fallback the extension shows "unavailable" and nothing else changes.

### Relevance evaluation (15 labeled pages, 2 study topics)

| Embeddings | Accuracy | False blocks | False allows |
|---|---|---|---|
| Lexical fallback | 73% (11/15) | 3 | 0 |
| sentence-transformers | 93% (14/15) | 0 | 0 |

With sentence-transformers, precision / recall / F1 were 0.83 / 1.00 / 0.91 for relevant, 1.00 / 0.75 / 0.86 for partially relevant and 1.00 / 1.00 / 1.00 for distracting content.

### Focus decay model

On synthetic test data the LSTM beats the "focus stays the same" baseline at every future window (mean absolute error 5.6 to 9.2 points versus 6.2 to 11.0).

## ⚠️ Known Limitations

- The labeled evaluation set is small (15 pages) and was written by the development team, so results are indicative only.
- The LSTM has been trained and tested on **synthetic** data. Its error numbers describe the simulator, not real students. It should be retrained and validated on real anonymized activity (excluding demo accounts) before being relied on.
- There is no login rate limiting, and a teacher can add any student to a cohort without that student's consent.
- Login tokens are kept in browser local storage.
- The semantic model needs about 2 GB of packages, which limits low-cost hosting options.
- The system has not been checked across many topics or languages.

## 🛣️ Future Work

- Retrain and validate the focus-decay model on real data, and explore Integrated Gradients or LRP on any classifier we train ourselves.
- Grow the labeled evaluation set with examples written by people who did not build the system.
- Add login rate limiting and student consent for cohort membership.
- Add support for more content platforms with dedicated extraction.
- Deploy the web app and backend publicly (not done yet).

## 💡 Why FocusAI?

FocusAI explores a practical question: **can a learning tool judge content by its relevance to a learner's current goal instead of judging it only by the website it appears on?**

By combining a browser extension, a semantic relevance engine, selective contextual reasoning, explainable decisions and learning analytics, the project brings together full-stack development, API design, browser integration and applied machine learning.

---

<div align="center">

### FocusAI

**Stay on topic. Learn with purpose.**

Built as a context-aware learning assistant.

</div>