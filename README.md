# LitQATrain

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/LitQATrain)

## Description

LitQATrain is an environment for evaluating scientific literature question answering with web search capabilities, inspired by FutureHouse's [LitQA2](https://github.com/Future-House/LitQA) benchmark. Like LitQA2, each question asks about a specific verifiable fact found in a particular scientific paper, and is designed to be specific enough that it can only be answered from a single source. LitQATrain extends this approach to 984 QA pairs across 10 broad scientific domains, with open-ended (non-multiple-choice) answers and web search tools for retrieval.

## Capabilities

- Scientific question answering across diverse domains
- Web search and information retrieval from academic literature
- Multi-step research: searching, reading papers, and synthesizing answers
- Verifiable factual recall from published research

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There is one split: **train** with 984 tasks. Questions span 10 scientific domains:

| Domain | Count |
|--------|-------|
| Molecular biology / Genomics | 100 |
| Neuroscience | 100 |
| Ecology / Environmental science | 100 |
| Chemistry / Materials science | 100 |
| Physics / Astronomy | 100 |
| Computer science / AI | 100 |
| Medicine / Clinical research | 100 |
| Earth science / Geology | 100 |
| Pharmacology / Drug development | 100 |
| Engineering / Applied science | 84 |

Each task provides a question and metadata (source DOI, domain). The agent prompt contains only the question; the agent must find the answer through web search.

## Reward Structure

This is a multi-turn environment. Agents use `web_search` and `fetch_url` tools to gather information, then submit via `submit_answer`. An LLM grader (gpt-5-mini) evaluates semantic equivalence between the submitted answer and the reference answer, handling synonyms, abbreviations, and equivalent scientific terminology. Reward is binary: 1.0 if correct, 0.0 if incorrect.

We do not use LLM graders from a different family for this task.

## Data

Data consists of a single Parquet file (`train.parquet`) containing QA pairs generated from scientific papers published in trusted journals (Nature, Science, PNAS, Cell, PLOS, ACS, arXiv, IEEE, AGU). Each row contains a question, answer, source DOI, key passage, and domain. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web using Tavily API. Returns up to 5 results with titles, URLs, and snippets. |
| `fetch_url` | Fetch full text content from a specific URL (truncated at 8000 characters). |
| `submit_answer` | Submit your final answer for LLM grading. Ends the episode. |

## Time Horizon

Multi-turn. Agents can perform multiple web searches and URL fetches before submitting a final answer.

## Environment Difficulty

[To be determined]

## Other Environment Requirements

- OpenAI API key required for LLM-based grading. Pass via `secrets={"openai_api_key": "..."}`.
- Tavily API key required for web search and URL fetching. Pass via `secrets={"tavily_api_key": "..."}`.

## Safety

Agents in LitQATrain answer scientific questions using web search in a standard environment. The environment does not present direct safety risks.

## Citations

This environment is inspired by LitQA2 from [FutureHouse's LAB-Bench](https://github.com/Future-House/LAB-Bench). Please cite the original work:

```bibtex
@article{laurent2024labbench,
  title     = {LAB-Bench: Measuring Capabilities of Language Models for Biology Research},
  author    = {Laurent, Jon M. and Janizek, Joseph D. and Ruzo, Michael and Hinks, Michaela M. and Hammerling, Michael J. and Narayanan, Siddharth and Ponnapati, Manvitha and White, Andrew D. and Rodriques, Samuel G.},
  year      = {2024},
  eprint    = {2407.10362},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

```bibtex
@dataset{GRLitQATrain,
  author    = {General Reasoning Inc. Team},
  title     = {LitQATrain},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/litqatrain}
}
```
