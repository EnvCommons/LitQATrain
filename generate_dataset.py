"""
LitQATrain Dataset Generator

Generates scientific literature QA pairs by:
1. Searching trusted academic sources for recent papers
2. Extracting content from paper URLs
3. Using LLM to generate verifiable QA pairs from paper content
4. Validating and deduplicating results

Usage:
    export OPENAI_API_KEY="sk-..."
    export TAVILY_API_KEY="tvly-..."
    python generate_dataset.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

# ============= Configuration =============

TARGET_PER_DOMAIN = 100  # 100 per domain = 1000 total.
OUTPUT_PARQUET = Path(__file__).parent / "train.parquet"
PROGRESS_PARQUET = Path(__file__).parent / "litqatrain_progress.parquet"
REFERENCE_FILE = Path(__file__).parent / "reference.txt"

DOMAINS = [
    "Molecular biology / Genomics",
    "Neuroscience",
    "Ecology / Environmental science",
    "Chemistry / Materials science",
    "Physics / Astronomy",
    "Computer science / AI",
    "Medicine / Clinical research",
    "Earth science / Geology",
    "Pharmacology / Drug development",
    "Engineering / Applied science",
]

# Trusted sources mapped to domains for targeted searches
# Many queries per domain to ensure enough extractable papers for 100 QA pairs each
DOMAIN_SEARCH_QUERIES = {
    "Molecular biology / Genomics": [
        "site:nature.com genomics gene regulation 2024 research",
        "site:nature.com CRISPR genome editing 2024 findings",
        "site:nature.com epigenetics chromatin 2024",
        "site:nature.com single cell RNA sequencing 2024",
        "site:nature.com transcription factor binding 2024",
        "site:journals.plos.org genetics molecular biology 2024",
        "site:journals.plos.org gene expression regulation 2024",
        "site:journals.plos.org CRISPR knockout screen 2024",
        "site:ncbi.nlm.nih.gov/pmc genomics sequencing 2024",
        "site:ncbi.nlm.nih.gov/pmc RNA splicing 2024",
        "site:ncbi.nlm.nih.gov/pmc whole genome sequencing 2024",
        "site:nature.com proteomics mass spectrometry 2024",
        "site:nature.com long read sequencing nanopore 2024",
        "site:journals.plos.org metagenomics microbiome 2024",
        "site:nature.com enhancer promoter interaction 2024",
        "site:nature.com spatial transcriptomics 2024",
        "site:journals.plos.org genome assembly annotation 2024",
        "site:nature.com DNA methylation epigenome 2024",
        "site:nature.com ribosome profiling translation 2024",
        "site:journals.plos.org population genetics 2024",
    ],
    "Neuroscience": [
        "site:nature.com neuroscience brain neural circuits 2024",
        "site:nature.com optogenetics neural activity 2024",
        "site:nature.com hippocampus memory formation 2024",
        "site:nature.com dopamine reward circuitry 2024",
        "site:nature.com cortical neurons electrophysiology 2024",
        "site:journals.plos.org neuroscience brain imaging 2024",
        "site:journals.plos.org synaptic plasticity LTP 2024",
        "site:journals.plos.org fMRI resting state connectivity 2024",
        "site:nature.com glia astrocyte microglia 2024",
        "site:nature.com pain nociception spinal cord 2024",
        "site:nature.com connectome brain mapping 2024",
        "site:journals.plos.org EEG neural oscillations 2024",
        "site:nature.com neurodegeneration tau amyloid 2024",
        "site:nature.com cerebellum motor learning 2024",
        "site:journals.plos.org olfaction taste sensory 2024",
        "site:nature.com retina visual processing 2024",
        "site:nature.com prefrontal cortex decision making 2024",
        "site:journals.plos.org axon regeneration injury 2024",
        "site:nature.com sleep circadian rhythm 2024",
        "site:nature.com basal ganglia movement disorders 2024",
    ],
    "Ecology / Environmental science": [
        "site:nature.com ecology biodiversity climate change 2024",
        "site:nature.com species extinction conservation 2024",
        "site:nature.com coral reef bleaching ocean 2024",
        "site:nature.com deforestation tropical forest 2024",
        "site:journals.plos.org ecology population dynamics 2024",
        "site:journals.plos.org invasive species impact 2024",
        "site:journals.plos.org marine ecology fisheries 2024",
        "site:journals.plos.org pollinator decline bees 2024",
        "site:nature.com carbon sequestration soil 2024",
        "site:nature.com wildfire ecosystem recovery 2024",
        "site:nature.com permafrost thaw methane 2024",
        "site:journals.plos.org bird migration phenology 2024",
        "site:journals.plos.org freshwater ecology rivers 2024",
        "site:nature.com microplastics environmental pollution 2024",
        "site:nature.com rewilding ecosystem restoration 2024",
        "site:journals.plos.org savanna grassland ecology 2024",
        "site:nature.com ocean acidification marine life 2024",
        "site:journals.plos.org amphibian decline chytrid 2024",
        "site:nature.com mangrove wetland ecosystem 2024",
        "site:journals.plos.org food web trophic interactions 2024",
    ],
    "Chemistry / Materials science": [
        "site:nature.com materials science catalysis 2024",
        "site:nature.com perovskite solar cell efficiency 2024",
        "site:nature.com battery lithium ion electrolyte 2024",
        "site:nature.com superconductor materials 2024",
        "site:pubs.acs.org chemistry synthesis 2024",
        "site:pubs.acs.org organic chemistry reaction 2024",
        "site:pubs.acs.org electrochemistry CO2 reduction 2024",
        "site:pubs.acs.org polymer nanoparticle 2024",
        "site:nature.com MOF metal organic framework 2024",
        "site:nature.com 2D materials graphene 2024",
        "site:pubs.acs.org photocatalysis hydrogen 2024",
        "site:nature.com semiconductor quantum dot 2024",
        "site:pubs.acs.org biomaterials hydrogel scaffold 2024",
        "site:nature.com topological insulator 2024",
        "site:pubs.acs.org flow chemistry continuous 2024",
        "site:nature.com ceramic composite high temperature 2024",
        "site:pubs.acs.org fluorescent probe sensor 2024",
        "site:nature.com thermoelectric materials 2024",
        "site:pubs.acs.org asymmetric catalysis enantioselective 2024",
        "site:nature.com covalent organic framework COF 2024",
        "site:nature.com nature chemistry organic synthesis 2024",
        "site:nature.com nature materials polymer 2024",
        "site:nature.com nature energy solar fuel 2024",
        "site:journals.plos.org chemistry nanoparticle 2024",
        "site:journals.plos.org materials characterization 2024",
        "site:nature.com catalyst selectivity mechanism 2024",
        "site:nature.com electrochemical water splitting 2024",
        "site:nature.com zeolite porous materials 2024",
        "site:nature.com nanostructure self-assembly 2024",
        "site:journals.plos.org spectroscopy analysis 2024",
        "site:nature.com alloy metallurgy strength 2024",
        "site:nature.com membrane separation gas 2024",
        "site:nature.com crystal structure determination 2024",
        "site:nature.com polymer electrolyte solid state 2024",
        "site:nature.com photovoltaic tandem efficiency 2024",
    ],
    "Physics / Astronomy": [
        "site:nature.com physics quantum computing qubit 2024",
        "site:nature.com gravitational waves LIGO 2024",
        "site:nature.com dark matter detection 2024",
        "site:nature.com exoplanet atmosphere JWST 2024",
        "site:arxiv.org quantum entanglement Bell 2024",
        "site:arxiv.org topological quantum 2024",
        "site:nature.com Bose-Einstein condensate ultracold 2024",
        "site:nature.com photonics integrated circuit 2024",
        "site:arxiv.org cosmology CMB 2024",
        "site:nature.com neutron star magnetar 2024",
        "site:nature.com laser attosecond ultrafast 2024",
        "site:arxiv.org string theory holographic 2024",
        "site:nature.com black hole event horizon 2024",
        "site:nature.com plasma fusion tokamak 2024",
        "site:arxiv.org particle physics collider 2024",
        "site:nature.com superconducting quantum processor 2024",
        "site:nature.com galaxy formation survey 2024",
        "site:arxiv.org quantum error correction 2024",
        "site:nature.com spin qubit silicon 2024",
        "site:nature.com muon anomalous magnetic moment 2024",
        "site:nature.com nature astronomy exoplanet 2024",
        "site:nature.com nature physics experiment 2024",
        "site:nature.com quantum simulation trapped ion 2024",
        "site:nature.com optical lattice cold atoms 2024",
        "site:nature.com gamma ray burst 2024",
        "site:journals.plos.org astrophysics stellar 2024",
        "site:nature.com supernova remnant 2024",
        "site:nature.com quantum sensing magnetometry 2024",
        "site:nature.com nanophotonics plasmonics 2024",
        "site:nature.com cosmic ray detection 2024",
        "site:nature.com topological photonics 2024",
        "site:nature.com nuclear physics isotope 2024",
        "site:journals.plos.org physics simulation model 2024",
        "site:nature.com metamaterial electromagnetic 2024",
        "site:nature.com spintronics magnetic 2024",
    ],
    "Computer science / AI": [
        "site:arxiv.org machine learning transformer architecture 2024",
        "site:arxiv.org large language model benchmark 2024",
        "site:arxiv.org reinforcement learning robotics 2024",
        "site:arxiv.org diffusion model image generation 2024",
        "site:arxiv.org graph neural network 2024",
        "site:nature.com artificial intelligence protein structure 2024",
        "site:nature.com AI drug discovery 2024",
        "site:arxiv.org federated learning privacy 2024",
        "site:arxiv.org vision language model multimodal 2024",
        "site:arxiv.org code generation programming 2024",
        "site:nature.com neural network interpretability 2024",
        "site:arxiv.org speech recognition whisper 2024",
        "site:arxiv.org adversarial robustness attack 2024",
        "site:arxiv.org knowledge graph embedding 2024",
        "site:nature.com AI weather prediction forecast 2024",
        "site:arxiv.org retrieval augmented generation RAG 2024",
        "site:arxiv.org object detection segmentation 2024",
        "site:nature.com AI materials discovery 2024",
        "site:arxiv.org time series forecasting 2024",
        "site:arxiv.org natural language processing NLP 2024",
        "site:nature.com machine learning genomics 2024",
        "site:nature.com deep learning medical imaging 2024",
        "site:nature.com AI climate model prediction 2024",
        "site:nature.com robot learning manipulation 2024",
        "site:journals.plos.org machine learning classification 2024",
        "site:journals.plos.org deep learning detection 2024",
        "site:journals.plos.org NLP text mining biomedical 2024",
        "site:nature.com computer vision autonomous 2024",
        "site:nature.com generative model sampling 2024",
        "site:journals.plos.org neural network prediction 2024",
        "site:nature.com foundation model pretraining 2024",
        "site:nature.com GNN molecular property 2024",
        "site:journals.plos.org random forest gradient boosting 2024",
        "site:nature.com attention mechanism self-supervised 2024",
        "site:nature.com AI scientific discovery 2024",
    ],
    "Medicine / Clinical research": [
        "site:nature.com clinical trial cancer immunotherapy 2024",
        "site:nature.com CAR-T cell therapy 2024",
        "site:nature.com vaccine mRNA efficacy 2024",
        "site:nature.com diabetes insulin treatment 2024",
        "site:nature.com Alzheimer disease biomarker 2024",
        "site:journals.plos.org clinical trial randomized 2024",
        "site:journals.plos.org surgery outcome survival 2024",
        "site:journals.plos.org epidemiology cohort study 2024",
        "site:nature.com gene therapy rare disease 2024",
        "site:nature.com antibiotic resistance MRSA 2024",
        "site:journals.plos.org tuberculosis treatment 2024",
        "site:nature.com cardiac heart failure therapy 2024",
        "site:journals.plos.org stroke rehabilitation 2024",
        "site:nature.com organ transplant rejection 2024",
        "site:nature.com obesity GLP-1 semaglutide 2024",
        "site:journals.plos.org maternal health pregnancy 2024",
        "site:nature.com autoimmune lupus rheumatoid 2024",
        "site:journals.plos.org malaria prevention treatment 2024",
        "site:nature.com sepsis critical care ICU 2024",
        "site:journals.plos.org HIV prevention PrEP 2024",
    ],
    "Earth science / Geology": [
        "site:nature.com earth science geoscience 2024",
        "site:nature.com earthquake fault seismology 2024",
        "site:nature.com volcanic eruption magma 2024",
        "site:nature.com ice sheet glacier melting 2024",
        "site:nature.com ocean circulation thermohaline 2024",
        "site:nature.com mineral formation crystal 2024",
        "site:nature.com paleoclimate proxy record 2024",
        "site:nature.com plate tectonics subduction 2024",
        "site:nature.com groundwater aquifer depletion 2024",
        "site:nature.com Mars geology rover 2024",
        "site:journals.plos.org geomorphology erosion 2024",
        "site:nature.com monsoon precipitation variability 2024",
        "site:nature.com tsunami coastal hazard 2024",
        "site:nature.com deep mantle plume 2024",
        "site:nature.com sediment stratigraphy basin 2024",
        "site:nature.com sea level rise coastal 2024",
        "site:journals.plos.org soil carbon organic 2024",
        "site:nature.com atmospheric chemistry aerosol 2024",
        "site:nature.com asteroid impact crater 2024",
        "site:nature.com geothermal energy subsurface 2024",
    ],
    "Pharmacology / Drug development": [
        "site:nature.com drug discovery pharmacology 2024",
        "site:nature.com antibody drug conjugate ADC 2024",
        "site:nature.com kinase inhibitor cancer 2024",
        "site:nature.com PROTAC targeted degradation 2024",
        "site:journals.plos.org pharmacokinetics bioavailability 2024",
        "site:journals.plos.org antimicrobial peptide 2024",
        "site:nature.com GPCR receptor agonist 2024",
        "site:nature.com RNA therapeutics antisense 2024",
        "site:journals.plos.org natural product drug 2024",
        "site:nature.com nanomedicine drug delivery 2024",
        "site:journals.plos.org toxicology adverse effects 2024",
        "site:nature.com epigenetic drug HDAC 2024",
        "site:nature.com pain analgesic opioid 2024",
        "site:journals.plos.org antiviral hepatitis 2024",
        "site:nature.com immunosuppressant transplant 2024",
        "site:journals.plos.org herbal medicine extract 2024",
        "site:nature.com bispecific antibody 2024",
        "site:nature.com allosteric modulator 2024",
        "site:journals.plos.org drug resistance mechanism 2024",
        "site:nature.com CRISPR therapeutic gene editing 2024",
    ],
    "Engineering / Applied science": [
        "site:nature.com engineering robotics soft actuator 2024",
        "site:nature.com biomedical engineering implant 2024",
        "site:nature.com microfluidics lab on chip 2024",
        "site:nature.com 3D printing additive manufacturing 2024",
        "site:nature.com wearable sensor health monitoring 2024",
        "site:nature.com water purification desalination 2024",
        "site:nature.com energy harvesting piezoelectric 2024",
        "site:nature.com autonomous vehicle navigation 2024",
        "site:journals.plos.org prosthetics rehabilitation 2024",
        "site:nature.com drone UAV remote sensing 2024",
        "site:nature.com tissue engineering organoid 2024",
        "site:journals.plos.org structural engineering earthquake 2024",
        "site:nature.com brain computer interface BCI 2024",
        "site:nature.com MEMS sensor accelerometer 2024",
        "site:nature.com optical fiber communication 2024",
        "site:journals.plos.org agricultural engineering irrigation 2024",
        "site:nature.com metamaterial acoustic cloaking 2024",
        "site:nature.com fuel cell hydrogen membrane 2024",
        "site:nature.com flexible electronics stretchable 2024",
        "site:journals.plos.org environmental monitoring IoT 2024",
    ],
}

# Trusted DOI prefixes
TRUSTED_DOI_PREFIXES = [
    "10.1038",   # Nature
    "10.1126",   # Science
    "10.1073",   # PNAS
    "10.1016",   # Elsevier / Cell Press
    "10.1128",   # ASM
    "10.1109",   # IEEE
    "10.1371",   # PLOS
    "10.48550",  # arXiv
    "10.1021",   # ACS
    "10.1002",   # Wiley
    "10.1146",   # Annual Reviews
    "10.1093",   # Oxford University Press
    "10.1103",   # APS (Physical Review)
    "10.1098",   # Royal Society
    "10.7554",   # eLife
    "10.1186",   # BMC / Springer
    "10.3389",   # Frontiers
    "10.1080",   # Taylor & Francis
    "10.1177",   # SAGE
    "10.1111",   # Wiley
    "10.1039",   # RSC
    "10.1088",   # IOP
    "10.1007",   # Springer
    "10.1029",   # AGU
]


# ============= Pydantic Models =============

class QAPair(BaseModel):
    question: str = Field(..., description="A verifiable factual question about a specific finding")
    answer: str = Field(..., description="A short, precise answer (under 50 characters)")
    source_doi: str = Field(..., description="The DOI of the source paper")
    key_passage: str = Field(..., description="The exact passage from the paper supporting the answer")
    domain: str = Field(..., description="The scientific domain")


# ============= Reference Examples =============

def load_reference_examples() -> str:
    with open(REFERENCE_FILE, "r") as f:
        examples = json.load(f)

    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"""Example {i}:
Question: {ex['question']}
Answer: {ex['answer']}
Source DOI: {ex['source_doi']}
Key passage: {ex['key_passage']}
Domain: {ex['domain']}""")

    return "\n\n".join(parts)


# ============= Extraction Helpers =============

DOI_PATTERN = re.compile(r'(10\.\d{4,9}/[^\s,;"\'>]+)')


def extract_dois_from_text(text: str) -> list[str]:
    """Extract DOI strings from text content."""
    matches = DOI_PATTERN.findall(text)
    # Clean trailing punctuation
    cleaned = []
    for m in matches:
        m = m.rstrip(".")
        if any(m.startswith(prefix) for prefix in TRUSTED_DOI_PREFIXES):
            cleaned.append(m)
    return list(set(cleaned))


def is_trusted_doi(doi: str) -> bool:
    """Check if a DOI is from a trusted publisher."""
    return any(doi.startswith(prefix) for prefix in TRUSTED_DOI_PREFIXES)


# ============= Core Pipeline =============

async def search_papers_for_domain(
    tavily_client: AsyncTavilyClient,
    domain: str,
    num_papers: int,
) -> list[dict]:
    """Search for papers in a specific domain using Tavily."""
    queries = DOMAIN_SEARCH_QUERIES.get(domain, [])
    all_results = []
    seen_urls = set()

    for query in queries:
        if len(all_results) >= num_papers * 3:  # Get extra for filtering
            break
        try:
            response = await tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=5,
            )
            results = response.get("results", [])
            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        "url": url,
                        "title": r.get("title", ""),
                        "snippet": r.get("content", ""),
                        "domain": domain,
                    })
            await asyncio.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"  Search error for '{query}': {e}")
            continue

    return all_results[:num_papers * 3]


async def fetch_paper_content(
    tavily_client: AsyncTavilyClient,
    url: str,
) -> str | None:
    """Fetch content from a paper URL using Tavily extract."""
    try:
        response = await tavily_client.extract(urls=[url])
        results = response.get("results", [])
        if not results:
            return None
        raw_content = results[0].get("raw_content", "")
        if len(raw_content) < 200:
            return None
        # Truncate very long content
        if len(raw_content) > 12000:
            raw_content = raw_content[:12000]
        return raw_content
    except Exception as e:
        print(f"  Fetch error for {url}: {e}")
        return None


async def generate_qa_from_content(
    oai_client: AsyncOpenAI,
    content: str,
    url: str,
    domain: str,
    reference_examples: str,
) -> QAPair | None:
    """Use LLM to generate a QA pair from paper content."""
    prompt = f"""You are creating a scientific literature question-answering benchmark. Given the content of a scientific paper, generate ONE high-quality question-answer pair.

Requirements for the question:
1. The answer must be a SPECIFIC, VERIFIABLE FACT from the paper (a number, percentage, fold-change, specific name, measurement, etc.)
2. The question should be answerable by finding the right paper, but NOT obvious without it
3. The question should ask about a specific finding or result, not general knowledge
4. The answer should be SHORT and PRECISE (under 50 characters)
5. You must extract the EXACT passage from the paper that contains the answer

CRITICAL - Question specificity:
The question MUST contain enough distinctive detail to uniquely identify the paper it comes from. Include specific terms like organism names, gene names, method names, named datasets, unique experimental setups, or distinctive technical terms that would appear in only this paper.

BAD example: "How many human organs were analyzed to explore vascular cell heterogeneity?"
- Too generic. Many papers study vascular cells across organs. Nothing pins this to one paper.

GOOD example: "Acinetobacter lwoffii has been evolved in the lab to be resistant to which antibiotic?"
- The specific organism name "Acinetobacter lwoffii" and the context "evolved in the lab" uniquely identify this paper.

GOOD example: "By what fold change do active olfactory receptor genes increase their contacts with Greek Island regions in mouse olfactory neurons?"
- "Greek Island regions" is a distinctive term unique to this line of research, making the question unambiguous.

Think: if someone searched the web for key terms in the question, would they find THIS paper specifically? If the question could apply to many papers, it is too generic. Add a distinctive identifier.

Here are examples of the style and quality we need:

{reference_examples}

Now generate a QA pair from this paper content:

URL: {url}
Domain: {domain}

Paper content:
{content}

Respond with a JSON object with these exact fields:
- "question": the question (string)
- "answer": short precise answer (string, under 50 chars)
- "source_doi": the DOI if found in the content, otherwise construct from the URL (string starting with "https://doi.org/")
- "key_passage": the exact verbatim passage from the paper that supports the answer (string)

IMPORTANT:
- The key_passage MUST be copied verbatim from the paper content above
- The answer must appear in or be directly derivable from the key_passage
- Pick a fact that is SPECIFIC to this paper's findings, not general domain knowledge
- If you cannot find a suitable specific fact, respond with {{"error": "no suitable fact found"}}"""

    try:
        response = await oai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result_text = response.choices[0].message.content or ""
        result = json.loads(result_text)

        if "error" in result:
            return None

        # Validate fields exist
        if not all(k in result for k in ["question", "answer", "source_doi", "key_passage"]):
            return None

        # Validate answer length
        if len(result["answer"]) > 50:
            return None

        # Validate DOI is from trusted source
        doi = result["source_doi"]
        if doi.startswith("https://doi.org/"):
            doi_id = doi[len("https://doi.org/"):]
        elif doi.startswith("http://doi.org/"):
            doi_id = doi[len("http://doi.org/"):]
        else:
            doi_id = doi

        if not is_trusted_doi(doi_id):
            return None

        return QAPair(
            question=result["question"],
            answer=result["answer"],
            source_doi=doi if doi.startswith("https://doi.org/") else f"https://doi.org/{doi_id}",
            key_passage=result["key_passage"],
            domain=domain,
        )
    except Exception as e:
        print(f"  QA generation error: {e}")
        return None


async def process_domain(
    oai_client: AsyncOpenAI,
    tavily_client: AsyncTavilyClient,
    domain: str,
    target_count: int,
    reference_examples: str,
    existing_questions: set[str],
) -> list[QAPair]:
    """Process a single domain: search, fetch, generate QA pairs."""
    print(f"\n{'='*60}")
    print(f"Processing domain: {domain}")
    print(f"Target: {target_count} QA pairs")
    print(f"{'='*60}")

    # Step 1: Search for papers
    print(f"  Searching for papers...")
    paper_results = await search_papers_for_domain(tavily_client, domain, target_count)
    print(f"  Found {len(paper_results)} candidate URLs")

    qa_pairs = []
    processed = 0

    for paper in paper_results:
        if len(qa_pairs) >= target_count:
            break

        processed += 1
        url = paper["url"]
        print(f"  [{processed}] Fetching: {url[:80]}...")

        # Step 2: Fetch content
        content = await fetch_paper_content(tavily_client, url)
        if not content:
            print(f"    -> No content extracted, skipping")
            continue

        print(f"    -> Got {len(content)} chars of content")

        # Step 3: Generate QA
        qa = await generate_qa_from_content(
            oai_client, content, url, domain, reference_examples
        )
        if not qa:
            print(f"    -> Failed to generate QA, skipping")
            continue

        # Step 4: Dedup check
        if qa.question in existing_questions:
            print(f"    -> Duplicate question, skipping")
            continue

        existing_questions.add(qa.question)
        qa_pairs.append(qa)
        print(f"    -> Generated QA #{len(qa_pairs)}: {qa.question[:60]}...")
        print(f"       Answer: {qa.answer}")

        await asyncio.sleep(1)  # Rate limiting

    print(f"  Domain complete: {len(qa_pairs)}/{target_count} QA pairs generated")
    return qa_pairs


def save_progress(qa_pairs: list[QAPair], path: Path) -> None:
    """Save current progress to parquet."""
    records = [
        {
            "question": qa.question,
            "answer": qa.answer,
            "source_doi": qa.source_doi,
            "key_passage": qa.key_passage,
            "domain": qa.domain,
        }
        for qa in qa_pairs
    ]
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    print(f"  Progress saved: {len(df)} records -> {path}")


def load_progress(path: Path) -> list[QAPair]:
    """Load previous progress from parquet."""
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return [
        QAPair(
            question=row["question"],
            answer=row["answer"],
            source_doi=row["source_doi"],
            key_passage=row["key_passage"],
            domain=row["domain"],
        )
        for _, row in df.iterrows()
    ]


async def main():
    # Validate API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not openai_api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    if not tavily_api_key:
        print("ERROR: Set TAVILY_API_KEY environment variable")
        sys.exit(1)

    oai_client = AsyncOpenAI(api_key=openai_api_key)
    tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

    # Load reference examples
    reference_examples = load_reference_examples()

    # Load any previous progress
    all_qa_pairs = load_progress(PROGRESS_PARQUET)
    existing_questions = {qa.question for qa in all_qa_pairs}

    if all_qa_pairs:
        print(f"Loaded {len(all_qa_pairs)} existing QA pairs from progress file")
        # Count per domain
        domain_counts = {}
        for qa in all_qa_pairs:
            domain_counts[qa.domain] = domain_counts.get(qa.domain, 0) + 1
        for d, c in sorted(domain_counts.items()):
            print(f"  {d}: {c}")

    # Process each domain
    for domain in DOMAINS:
        # Count existing for this domain
        existing_count = sum(1 for qa in all_qa_pairs if qa.domain == domain)
        remaining = TARGET_PER_DOMAIN - existing_count

        if remaining <= 0:
            print(f"\nSkipping {domain} (already have {existing_count}/{TARGET_PER_DOMAIN})")
            continue

        domain_pairs = await process_domain(
            oai_client, tavily_client, domain, remaining, reference_examples, existing_questions
        )
        all_qa_pairs.extend(domain_pairs)

        # Save progress after each domain
        save_progress(all_qa_pairs, PROGRESS_PARQUET)

    # Final output
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total QA pairs: {len(all_qa_pairs)}")

    # Add IDs and save final output
    records = []
    for idx, qa in enumerate(all_qa_pairs):
        records.append({
            "id": f"litqatrain_train_{idx}",
            "question": qa.question,
            "answer": qa.answer,
            "source_doi": qa.source_doi,
            "key_passage": qa.key_passage,
            "domain": qa.domain,
        })

    df = pd.DataFrame(records)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved to {OUTPUT_PARQUET}")

    # Print domain distribution
    print(f"\nDomain distribution:")
    for domain, count in df["domain"].value_counts().items():
        print(f"  {domain}: {count}")

    # Print answer length stats
    print(f"\nAnswer length stats:")
    print(df["answer"].str.len().describe())

    # Print sample
    print(f"\nSample QA pairs:")
    for _, row in df.head(3).iterrows():
        print(f"\n  Q: {row['question']}")
        print(f"  A: {row['answer']}")
        print(f"  DOI: {row['source_doi']}")
        print(f"  Domain: {row['domain']}")


if __name__ == "__main__":
    asyncio.run(main())
