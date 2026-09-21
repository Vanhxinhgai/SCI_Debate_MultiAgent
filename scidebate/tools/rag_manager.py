"""RAG Manager for Scientific Literature.

Generates optimized search queries using an optional LLM or a standard fallback,
queries arXiv, and compiles retrieved paper summaries into structured Markdown evidence.
"""
import re
from scidebate.llms import BaseLLM
from scidebate.tools.arxiv_search import search_arxiv
from scidebate.tools.scifact import retrieve_scifact_evidence

# Danh sách từ dừng để lọc từ khóa khi không có LLM
STOPWORDS = {
    "is", "are", "was", "were", "be", "been", "being",
    "the", "a", "an", "and", "or", "but", "if", "then",
    "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further",
    "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "would", "could", "may", "might", "must",
    "people", "human", "they", "we", "you", "them", "always", "never"
}

# ── Domain-aware filter: education/cognitive multitasking claims ──────────────

# Group A: multitasking concepts (the "X" in the claim)
_GROUP_A = {
    "multitasking", "multi-tasking", "media multitasking", "task switching",
    "task-switching", "dual task", "dual-task", "divided attention",
    "cognitive load", "concurrent tasks", "attention switching",
    "simultaneous tasks", "attention divided",
}

# Group B: education/learning outcomes (the "Y" in the claim)
_GROUP_B = {
    "academic performance", "academic achievement", "gpa", "grades", "exam",
    "test score", "learning outcomes", "student performance", "students",
    "classroom", "education", "school", "university", "study", "learning",
    "memory", "attention", "cognition", "comprehension",
}

# Domains completely unrelated to academic multitasking — penalize heavily
_UNRELATED_DOMAINS = {
    "neutrophil", "netos", "netosis", "software refactoring", "genetic algorithm",
    "job scheduling", "multiprocessor", "microsoft academic", "citation impact",
    "computational linguistics", "software model",
}

# Education + multitasking trigger terms in claims
_CLAIM_MULTITASK_TERMS = {
    "multitasking", "multi-tasking", "media multitasking", "task switching",
    "divided attention", "dual task", "cognitive load",
}
_CLAIM_EDU_TERMS = {
    "academic", "performance", "learning", "student", "students", "gpa",
    "grades", "exam", "education", "school", "university", "memory",
}


def _is_education_multitask_claim(claim: str) -> bool:
    """Detect if claim is about multitasking effects on academic/learning outcomes."""
    claim_lower = claim.lower()
    has_multitask = any(t in claim_lower for t in _CLAIM_MULTITASK_TERMS)
    has_edu = any(t in claim_lower for t in _CLAIM_EDU_TERMS)
    return has_multitask and has_edu


def _education_domain_query(stance: str) -> str:
    """Return a direct education-domain arXiv query."""
    if stance == "CON":
        return "media multitasking academic performance impair students learning"
    elif stance == "PRO":
        return "multitasking students academic performance cognitive learning"
    return "multitasking academic performance students learning outcomes"


def _paper_domain_score(paper: dict) -> int:
    """Score −3 to +3 for domain relevance to education/multitasking."""
    text = f"{paper.get('title', '')} {paper.get('summary', '')}".lower()
    score = 0
    if any(t in text for t in _GROUP_A):
        score += 2
    if any(t in text for t in _GROUP_B):
        score += 1
    if any(t in text for t in _UNRELATED_DOMAINS):
        score -= 3
    return score


def _filter_domain_papers(papers: list, claim: str) -> list:
    """Re-rank and filter papers for domain relevance when claim is edu+multitask."""
    if not papers or not _is_education_multitask_claim(claim):
        return papers

    scored = [(p, _paper_domain_score(p)) for p in papers]
    # Keep papers with positive score, sorted best-first
    relevant = sorted([(p, s) for p, s in scored if s > 0], key=lambda x: x[1], reverse=True)
    if relevant:
        return [p for p, _ in relevant]
    # Nothing passed — return all but still sorted (so worst unrelated are last)
    return [p for p, _ in sorted(scored, key=lambda x: x[1], reverse=True)]


def generate_search_query_simple(claim: str, stance: str = "NEUTRAL") -> str:
    """Domain-aware keyword extraction with education/multitasking shortcut."""
    if _is_education_multitask_claim(claim):
        return _education_domain_query(stance)

    words = re.findall(r'\b\w+\b', claim.lower())
    filtered_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    if not filtered_words:
        query = claim
    else:
        query = " ".join(filtered_words[:4])

    if stance == "CON":
        query += " impair harm negative"
    return query

def generate_search_query_llm(claim: str, llm: BaseLLM, stance: str = "NEUTRAL") -> str:
    """Sử dụng LLM sinh ra 2-3 từ khóa tìm kiếm tối ưu trên arXiv."""
    # Short-circuit for education+multitasking — LLM queries often retrieve wrong domains
    if _is_education_multitask_claim(claim):
        return _education_domain_query(stance)

    if stance == "PRO":
        stance_inst = "supporting or verifying the claim (look for positive trials, confirmation, correlation, or protective effects)"
    elif stance == "CON":
        stance_inst = "contradicting, refuting, or casting skepticism on the claim (look for limitations, contradictions, negative trials, or lack of correlation)"
    else:
        stance_inst = "relevant to the claim"

    prompt = f"""You are a scientific research assistant. Given the claim:
"{claim}"

Output ONLY a short search query (2 to 4 keywords) to find scientific papers {stance_inst} on arXiv.
Do NOT include any introduction, quotes, punctuation, or other text.
Optimal search query:"""
    try:
        response = llm.generate([{"role": "user", "content": prompt}], max_tokens=15, temperature=0.0)
        query = response.content.strip().strip('"').strip("'")
        if query and len(query) < 100:
            return query
    except Exception as e:
        print(f"Error generating search query with LLM: {e}")
    return generate_search_query_simple(claim, stance)

def retrieve_evidence(
    claim: str,
    max_results: int = 3,
    generator_llm: BaseLLM = None,
    stance: str = "NEUTRAL",
    source: str = "arxiv",
) -> dict:
    """Retrieve evidence from the configured source and compile RAG context."""
    source_key = (source or "arxiv").strip().lower()
    if source_key == "scifact":
        return retrieve_scifact_evidence(claim, max_results=max_results, stance=stance)
    if source_key in {"hybrid", "both", "scifact+arxiv", "arxiv+scifact"}:
        half = max(1, max_results // 2)
        scifact_res = retrieve_scifact_evidence(claim, max_results=half, stance=stance)
        arxiv_res = retrieve_evidence(
            claim,
            max_results=max_results - half,
            generator_llm=generator_llm,
            stance=stance,
            source="arxiv",
        )
        context = "\n\n".join([
            "### HYBRID EVIDENCE BASE (SCIFACT + ARXIV)",
            "Prioritize gold SciFact evidence when available. Use arXiv evidence only as supplementary context.",
            scifact_res.get("context", ""),
            arxiv_res.get("context", ""),
        ])
        return {
            "context": context,
            "papers": scifact_res.get("papers", []) + arxiv_res.get("papers", []),
            "search_query": {
                "scifact": scifact_res.get("search_query", claim),
                "arxiv": arxiv_res.get("search_query", claim),
            },
            "source": "hybrid",
            "scifact": scifact_res,
            "arxiv": arxiv_res,
        }

    """Tìm kiếm tài liệu trên arXiv và biên dịch thành ngữ cảnh RAG hoàn chỉnh."""
    if generator_llm:
        search_query = generate_search_query_llm(claim, generator_llm, stance)
    else:
        search_query = generate_search_query_simple(claim, stance)
        
    papers = search_arxiv(search_query, max_results)
    
    # Chỉ chạy fallback tìm kiếm theo claim nếu query đầu tiên thành công nhưng không có kết quả (trả về list rỗng [])
    # Nếu query đầu tiên thất bại/timeout (trả về None), ta không chạy fallback để tránh treo timeout tiếp.
    if isinstance(papers, list) and len(papers) == 0 and search_query != claim:
        fallback_papers = search_arxiv(claim, max_results)
        if isinstance(fallback_papers, list):
            papers = fallback_papers
            search_query = claim
        else:
            papers = []
    elif papers is None:
        papers = []

    # Domain filter + re-rank for education/multitasking claims
    papers = _filter_domain_papers(papers, claim)
        
    # Biên dịch tài liệu thành Markdown ngữ cảnh
    if not papers:
        context = f"No scientific publications found on arXiv directly addressing this claim with a {stance} stance."
    else:
        if stance == "PRO":
            header = "### SCIENTIFIC EVIDENCE BASE - SUPPORTING (PRO)"
        elif stance == "CON":
            header = "### SCIENTIFIC EVIDENCE BASE - OPPOSING (CON)"
        else:
            header = "### SCIENTIFIC EVIDENCE BASE (RETRIEVED FROM ARXIV)"

        context_lines = [
            header,
            "The following peer-reviewed publication abstracts are available as potential evidence.",
            "You may cite their authors and years ONLY if they are directly relevant to the claim. Do NOT force a connection if they are irrelevant. Combine relevant papers with your own internal scientific knowledge to state your case.",
            ""
        ]
        for i, paper in enumerate(papers, 1):
            context_lines.extend([
                f"Document [{i}]:",
                f"- Title: {paper['title']}",
                f"- Authors: {paper['authors']}",
                f"- Date: {paper['published']}",
                f"- PDF Link: {paper['pdf_link']}" if paper['pdf_link'] else "- PDF Link: N/A",
                f"- Abstract Summary: {paper['summary']}",
                ""
            ])
        context = "\n".join(context_lines)
        
    return {
        "context": context,
        "papers": papers,
        "search_query": search_query
    }
