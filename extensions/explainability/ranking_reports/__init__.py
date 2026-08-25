"""Per-user ranking reports: what the model recommended, next to what the user did.

Where ``decomposition`` explains a score from the model's own internals, this
reads the ranking from the outside: the top-K with real product titles, the
held-out ground truth, and the user's history in both domains, written as
markdown to read and JSON to feed to something else.

    extract_user_ranking           one user, the first of the test split
    extract_user_ranking_enhanced  a random sample of users, plus an aggregate
    core.ranking_payload           the JSON schema both of them emit
    run_llm_relevance              asks a local LLM which of the items counted
                                   as irrelevant plausibly are not

Paths in these scripts (``saved/``, the metadata dumps, ``ranking_logs/``) are
relative to the working directory, so run them from the repository root:

    python -m extensions.explainability.ranking_reports.extract_user_ranking
    python -m extensions.explainability.ranking_reports.run_llm_relevance --require-citation
"""
