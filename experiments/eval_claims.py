"""Test claim set with ground-truth verdicts for system evaluation.

20 claims covering 4 categories:
  S* — SUPPORTED   (strong scientific consensus)
  R* — REFUTED     (contradicted by evidence)
  I* — INCONCLUSIVE (mixed/limited evidence)
  C* — CORRELATIONAL (associational claims that should be SUPPORTED)

Ground truth is based on peer-reviewed scientific consensus as of 2024-2025.
References are included for each claim for verification.
"""

TEST_CLAIMS = [
    # ── SUPPORTED (5) ─────────────────────────────────────────
    {
        "id": "S01",
        "claim": "Sleep deprivation impairs memory consolidation.",
        "ground_truth": "SUPPORTED",
        "claim_type": "CAUSAL",
        "domain": "neuroscience",
        "difficulty": "easy",
        "reference": "Walker & Stickgold (2004) Science; Diekelmann & Born (2010) Nature Reviews Neuroscience",
        "notes": "Well-established RCT and mechanistic evidence; slow-wave and REM sleep roles confirmed.",
    },
    {
        "id": "S02",
        "claim": "Regular aerobic exercise reduces risk of type 2 diabetes.",
        "ground_truth": "SUPPORTED",
        "claim_type": "CAUSAL",
        "domain": "medicine",
        "difficulty": "easy",
        "reference": "Diabetes Prevention Program (2002) NEJM; meta-analysis Umpierre et al. (2011) JAMA",
        "notes": "Multiple large RCTs confirm causal reduction in insulin resistance.",
    },
    {
        "id": "S03",
        "claim": "Smoking causes lung cancer.",
        "ground_truth": "SUPPORTED",
        "claim_type": "CAUSAL",
        "domain": "oncology",
        "difficulty": "easy",
        "reference": "IARC Group 1 carcinogen; US Surgeon General 1964; Bradford Hill criteria fully met",
        "notes": "Gold standard causal evidence; 85-90% of lung cancers attributable to smoking.",
    },
    {
        "id": "S04",
        "claim": "Handwashing with soap reduces transmission of diarrhoeal disease.",
        "ground_truth": "SUPPORTED",
        "claim_type": "CAUSAL",
        "domain": "public_health",
        "difficulty": "easy",
        "reference": "Cochrane Review Curtis & Cairncross (2003); WHO guidelines",
        "notes": "Systematic reviews show 40-50% reduction; high-quality intervention studies.",
    },
    {
        "id": "S05",
        "claim": "Antiretroviral therapy reduces HIV viral load in patients.",
        "ground_truth": "SUPPORTED",
        "claim_type": "CAUSAL",
        "domain": "medicine",
        "difficulty": "easy",
        "reference": "Ho et al. (1995) Nature; PARTNER Study (2019) JAMA",
        "notes": "Direct mechanistic and clinical trial evidence; undetectable viral load achieved.",
    },

    # ── REFUTED (5) ───────────────────────────────────────────
    {
        "id": "R01",
        "claim": "Multitasking always improves academic performance.",
        "ground_truth": "REFUTED",
        "claim_type": "UNIVERSAL_CAUSAL",
        "domain": "education",
        "difficulty": "medium",
        "reference": "Ophir et al. (2009) PNAS; Junco & Cotten (2012) Computers & Education",
        "notes": "Universal claim contradicted by media-multitasking literature showing GPA decline; capacity-bound effects.",
    },
    {
        "id": "R02",
        "claim": "Vaccines cause autism.",
        "ground_truth": "REFUTED",
        "claim_type": "CAUSAL",
        "domain": "medicine",
        "difficulty": "easy",
        "reference": "Taylor et al. (1999) Lancet; Institute of Medicine (2004); retracted Wakefield (1998)",
        "notes": "Multiple large epidemiological studies found no link; original paper retracted for fraud.",
    },
    {
        "id": "R03",
        "claim": "Humans only use 10% of their brains.",
        "ground_truth": "REFUTED",
        "claim_type": "GENERAL",
        "domain": "neuroscience",
        "difficulty": "easy",
        "reference": "Neuroimaging studies; Barry Beyerstein (1999) Scientific American",
        "notes": "fMRI shows all brain regions active; no dormant 90% detected.",
    },
    {
        "id": "R04",
        "claim": "Antibiotics are effective against viral infections.",
        "ground_truth": "REFUTED",
        "claim_type": "CAUSAL",
        "domain": "medicine",
        "difficulty": "easy",
        "reference": "Mechanism of action: antibiotics target bacterial cell walls/ribosomes, not viral replication",
        "notes": "Mechanistic refutation; widely confirmed in microbiology.",
    },
    {
        "id": "R05",
        "claim": "Sugar consumption directly causes hyperactivity in children.",
        "ground_truth": "REFUTED",
        "claim_type": "CAUSAL",
        "domain": "medicine",
        "difficulty": "medium",
        "reference": "Wolraich et al. (1995) JAMA meta-analysis; double-blind RCT evidence",
        "notes": "Controlled trials show no effect; expectation bias documented in parents.",
    },

    # ── INCONCLUSIVE (5) ──────────────────────────────────────
    {
        "id": "I01",
        "claim": "Multitasking improves academic performance.",
        "ground_truth": "INCONCLUSIVE",
        "claim_type": "CAUSAL",
        "domain": "education",
        "difficulty": "medium",
        "reference": "Mixed literature: Junco (2015) shows negative effects; some studies show null effects",
        "notes": "Evidence predominantly negative; no RCT with positive outcome; verdict depends on multitasking type.",
    },
    {
        "id": "I02",
        "claim": "Vitamin C supplementation prevents the common cold.",
        "ground_truth": "INCONCLUSIVE",
        "claim_type": "CAUSAL",
        "domain": "medicine",
        "difficulty": "medium",
        "reference": "Cochrane Review Hemilä & Chalker (2013): reduces duration but not incidence",
        "notes": "Reduces duration slightly in general population; prevents in extreme physical stress. Mixed overall.",
    },
    {
        "id": "I03",
        "claim": "Intermittent fasting improves long-term cognitive function.",
        "ground_truth": "INCONCLUSIVE",
        "claim_type": "CAUSAL",
        "domain": "neuroscience",
        "difficulty": "hard",
        "reference": "de Cabo & Mattson (2019) NEJM; limited human RCT data",
        "notes": "Strong animal model evidence; insufficient long-term human RCT data.",
    },
    {
        "id": "I04",
        "claim": "Social media use causes depression in teenagers.",
        "ground_truth": "INCONCLUSIVE",
        "claim_type": "CAUSAL",
        "domain": "psychology",
        "difficulty": "hard",
        "reference": "Orben & Przybylski (2019) Nature Human Behaviour; Haidt & Allen (2020)",
        "notes": "Correlation established; causal direction debated; effect size small; confounders uncontrolled.",
    },
    {
        "id": "I05",
        "claim": "Eating breakfast improves academic performance in children.",
        "ground_truth": "INCONCLUSIVE",
        "claim_type": "CAUSAL",
        "domain": "education",
        "difficulty": "medium",
        "reference": "Hoyland et al. (2009) review; confounds: SES, nutrition quality, school type",
        "notes": "Short-term attention effects shown; long-term academic outcomes confounded by SES.",
    },

    # ── CORRELATIONAL (5) — should be SUPPORTED ───────────────
    {
        "id": "C01",
        "claim": "Media multitasking is associated with lower academic performance among students.",
        "ground_truth": "SUPPORTED",
        "claim_type": "CORRELATIONAL",
        "domain": "education",
        "difficulty": "medium",
        "reference": "Junco & Cotten (2012) C&E; Burak (2012); End et al. (2010)",
        "notes": "Consistent correlational evidence across multiple studies; appropriate claim for SUPPORTED.",
    },
    {
        "id": "C02",
        "claim": "Physical activity is associated with reduced symptoms of anxiety.",
        "ground_truth": "SUPPORTED",
        "claim_type": "CORRELATIONAL",
        "domain": "medicine",
        "difficulty": "easy",
        "reference": "Asmundson et al. (2013) meta-analysis; Stonerock et al. (2015)",
        "notes": "Strong correlational and some causal evidence; appropriate SUPPORTED verdict.",
    },
    {
        "id": "C03",
        "claim": "Socioeconomic status is associated with academic achievement.",
        "ground_truth": "SUPPORTED",
        "claim_type": "CORRELATIONAL",
        "domain": "education",
        "difficulty": "easy",
        "reference": "Sirin (2005) meta-analysis; PISA/TIMSS international data",
        "notes": "One of most replicated findings in educational research; correlation well established.",
    },
    {
        "id": "C04",
        "claim": "Blue light exposure before sleep is linked to disrupted sleep patterns.",
        "ground_truth": "SUPPORTED",
        "claim_type": "CORRELATIONAL",
        "domain": "medicine",
        "difficulty": "easy",
        "reference": "Chang et al. (2015) PNAS; Harvard Medical School studies",
        "notes": "Melatonin suppression mechanism identified; correlational and experimental evidence.",
    },
    {
        "id": "C05",
        "claim": "Higher educational attainment is associated with better health outcomes.",
        "ground_truth": "SUPPORTED",
        "claim_type": "CORRELATIONAL",
        "domain": "public_health",
        "difficulty": "easy",
        "reference": "Ross & Wu (1995); Cutler & Lleras-Muney (2010) Journal of Health Economics",
        "notes": "Robust across countries and time periods; health literacy, income, occupational pathways identified.",
    },
]

# Quick subset (5 claims) for fast testing — one from each category
QUICK_CLAIMS = [
    next(c for c in TEST_CLAIMS if c["id"] == "S01"),  # Sleep deprivation
    next(c for c in TEST_CLAIMS if c["id"] == "R01"),  # Multitasking always
    next(c for c in TEST_CLAIMS if c["id"] == "I02"),  # Vitamin C
    next(c for c in TEST_CLAIMS if c["id"] == "C01"),  # Media multitasking correlation
    next(c for c in TEST_CLAIMS if c["id"] == "R02"),  # Vaccines autism
]


def get_claims(mode: str = "full") -> list:
    if mode == "quick":
        return QUICK_CLAIMS
    if mode == "supported":
        return [c for c in TEST_CLAIMS if c["ground_truth"] == "SUPPORTED"]
    if mode == "refuted":
        return [c for c in TEST_CLAIMS if c["ground_truth"] == "REFUTED"]
    if mode == "inconclusive":
        return [c for c in TEST_CLAIMS if c["ground_truth"] == "INCONCLUSIVE"]
    return TEST_CLAIMS
