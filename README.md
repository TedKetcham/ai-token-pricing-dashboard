# AI Token Pricing Intelligence Dashboard

**MIS Group Assignment | AI-Powered Business Decision Support**

**[Launch the Live Dashboard](https://ai-token-pricing-dashboard.streamlit.app/)** | **[View GitHub Repository](https://github.com/jonathonglee/ai-token-pricing-dashboard)**

## Project Overview

AI companies offer different models with different pricing structures, making it difficult for businesses to estimate costs and compare providers. Our project addresses this challenge through an interactive, AI-powered dashboard that compares standard text-generation API token pricing across selected models from OpenAI, Google, and Anthropic.

### Business Question

**How can businesses compare AI model API costs, estimate spending under different usage scenarios, and use trustworthy research to support vendor selection?**

### Target Stakeholders

- **IT and AI project managers:** Evaluate API options for proposed applications.
- **Financial analysts:** Estimate monthly operating expenses and potential savings.
- **Procurement teams:** Compare providers using transparent pricing assumptions.
- **Business decision-makers:** Explore trade-offs before selecting an AI model.

## Key Features

### 1. Interactive Pricing Comparison

Users can filter AI providers and compare model pricing based on:

- Monthly API request volume
- Average input tokens per request
- Average output tokens per request
- Standard input and output token rates

The dashboard calculates estimated monthly costs and identifies the lowest-cost option within the selected dataset.

### 2. Business Case Analysis

Users can select a baseline model and compare potential monthly savings from switching to alternatives. The dashboard also explains why pricing alone should not determine the final vendor decision.

### 3. Retrieval-Augmented Generation (RAG) Research Assistant

The AI-powered research assistant answers pricing questions using retrieved excerpts from official provider documentation alongside the dashboard's stored pricing data.

The workflow consists of:

1. Retrieve official pricing documents.
2. Divide retrieved text into searchable passages.
3. Rank passages based on relevance to the user's question.
4. Provide relevant evidence and stored pricing data to an OpenAI model.
5. Generate an analytical response with supporting links and source timestamps.

The assistant identifies limitations when retrieved evidence does not independently substantiate a stored price.

### 4. Official Source Verification

The dashboard retrieves documentation from:

- [OpenAI API Pricing](https://developers.openai.com/api/docs/pricing)
- [Google Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Anthropic Claude Pricing](https://platform.claude.com/docs/en/about-claude/pricing)

An LLM-assisted extraction feature can propose candidate pricing updates. Proposed changes require human verification and approval before modifying the approved dataset.

### 5. Automated Research Workflow

GitHub Actions runs a scheduled workflow designed to retrieve official pricing documents and generate review reports.

**Successful manual test results:**

- Three of three official source documents retrieved.
- Refresh status report created.
- Pending pricing review report created.
- Reports successfully committed to GitHub.
- Zero candidate pricing changes identified during the test.

This demonstrates automated source collection and reporting, not autonomous verification or approval of new prices.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and research logic |
| Streamlit | Interactive dashboard |
| Pandas | Pricing calculations and data processing |
| Plotly | Interactive visualizations |
| OpenAI API | RAG answer generation and assisted extraction |
| GitHub Actions | Scheduled research workflow |
| GitHub | Version control and project hosting |
| Streamlit Community Cloud | Public application deployment |

## Demonstration Scenario

A business expects to process 100,000 API requests per month, averaging 1,000 input tokens and 500 output tokens per request.

Using the project's stored pricing snapshot, GPT-5 nano has the lowest estimated monthly token cost among the seven selected models, at $25.

This represents a potential monthly saving of $5 compared with Gemini 2.5 Flash-Lite and $100 compared with GPT-5 mini.

These estimates are based on token costs only and do not establish which model provides the best overall business value.

## Research Limitations

- The approved pricing dataset is a dated snapshot, not a continuously verified live price feed.
- Retrieved webpages may contain irrelevant information or ambiguous pricing tables.
- AI-assisted price extraction can fail to identify rates and requires human review.
- Scheduled collection does not automatically approve new prices.
- Only selected text-generation API models are compared.
- Model quality, response speed, security, context limits, caching, discounts, and other costs may affect procurement decisions.
- Public API documentation can change, so prices should be rechecked before purchasing.

## Run Locally

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Launch the dashboard:

```powershell
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

An OpenAI API key is required for AI-generated research answers. Store API keys securely using environment variables or Streamlit Secrets. Never commit real keys to GitHub.

## Live Application

**Dashboard:** https://ai-token-pricing-dashboard.streamlit.app/

**Source code:** https://github.com/jonathonglee/ai-token-pricing-dashboard

## Conclusion

This project demonstrates how interactive analytics, retrieval-augmented generation, and automated source collection can support AI procurement decisions. By combining cost calculations with research evidence and human review, the dashboard helps businesses explore AI model pricing while maintaining transparency about data reliability and research limitations.
