# AI Token Pricing Intelligence — MIS Group Assignment

A working interactive Streamlit dashboard comparing standard paid **text token API pricing** across OpenAI, Google, and Anthropic. Intended stakeholder: a business choosing a model provider and budgeting API usage.

## Features
- Provider filters and interactive token-based monthly cost comparisons.
- Procurement scenario and savings relative to a chosen baseline.
- **Working RAG**: fetch official pricing pages, chunk retrieved text, rank excerpts by query relevance, ask OpenAI to answer using only retrieved evidence, display citations and timestamps.
- **Assisted price verification**: fetch three official sources, use an LLM to propose candidate prices, show evidence and require human approval before writing the approved CSV.
- Daily GitHub Action fetches source pages and writes a **pending review report**; it does not silently change approved prices. If an API key is configured as a GitHub Actions secret, the scheduled job also proposes price updates.
- Source register, export, and real historical snapshots (when approved updates are recorded).

## Run on Windows
Open PowerShell inside the extracted folder containing `app.py`:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501. In the sidebar, enter your **class OpenAI API key** to use RAG and assisted extraction. **Never paste it in GitHub, a public README, or screenshots.** Alternatively, create `.streamlit/secrets.toml` locally from `.streamlit/secrets.toml.example` and replace the placeholder.

## Five-minute live demo
1. (0:00–0:40) Stakeholder problem: choosing an affordable LLM API.
2. (0:40–1:40) Change monthly requests and input/output token counts; compare costs.
3. (1:40–2:20) Explain potential savings and limitations of cost-only rankings.
4. (2:20–3:35) Open **Source verification**; fetch official pages, show timestamps and extraction proposals. Do not approve unverified proposals.
5. (3:35–4:40) Ask **RAG research assistant** a question; show retrieved citations.
6. (4:40–5:00) Explain scheduled GitHub Action and human approval safeguard.

## Deploy on Streamlit Community Cloud
1. Upload the project files (including `.github/workflows/refresh.yml`) to a public GitHub repository. Do **not** upload `.streamlit/secrets.toml` or your API key.
2. Go to https://share.streamlit.io, choose the repository and `app.py`, and deploy.
3. Under app settings **Secrets**, add `OPENAI_API_KEY = "your-secret-value"` if you want the professor to demonstrate AI without typing a key. Treat a publicly accessible AI-powered app as potentially billable: set API usage limits, restrict access if possible, or remove the key after grading.
4. In GitHub repository **Settings → Secrets and variables → Actions**, optionally add `OPENAI_API_KEY` for automated candidate extraction. This is separate from Streamlit secrets.
5. Submit the deployed `*.streamlit.app` URL and GitHub repository URL. Test in incognito.

## Research methodology and limitations
The initial `prices.csv` is a dated **approved research snapshot**, not live data. `verified_date` records the snapshot date. Official webpages can change; fetching alone does not verify numeric rates. The LLM extractor is deliberately conservative and **requires human review** before modifying the approved price table. Scheduled GitHub Actions creates a fresh status/pending-review report, but **does not autonomously publish new approved prices**. This is a deliberate integrity safeguard, not a claim of fully automated verified real-time pricing. On Streamlit Community Cloud, writes to the app filesystem are not reliably persistent; commit approved changes to GitHub for lasting deployment.

The dashboard focuses on comparable public text API token pricing. NVIDIA and Apple are not directly ranked because comparable first-party general-purpose text-token API prices are not established. Cost-only comparisons do not assess model quality, speed, latency, privacy, long-context pricing, caching, tool fees, discounts, taxes, or availability. Rates should be checked against official pages before the presentation.

Official pages: https://developers.openai.com/api/docs/pricing · https://ai.google.dev/gemini-api/docs/pricing · https://platform.claude.com/docs/en/about-claude/pricing

## Files
- `app.py` — Streamlit UI
- `research.py` — document retrieval, RAG, candidate extraction and approval
- `prices.csv` — dated pricing snapshot
- `scheduled_refresh.py` — scheduled fetch and pending review report
- `.github/workflows/refresh.yml` — daily scheduled research
- `.streamlit/secrets.toml.example` — safe template, not a real key
