# Benchmark and search-discovery audit — 18 October 2026

Scope: benchmark evidence and search/agent discovery for the current v0.14.0 code. No application runtime, model training, permissions, or execution behavior changed.

## Measured evidence

The fresh [software bundle](evidence/software-benchmarks-2026-10-18.json) was captured against source commit `481524e06c84e63c24b99651d031298b6688621b` on Windows 11 / Python 3.13.13. The JSON's full `source_commit` is authoritative.

- 100 guarded CPU-status iterations: 28.727 ms median, 32.192 ms p95, zero model calls.
- Four real read-only status scenarios, 50 measured iterations each.
- Deterministic routing: 59/59 fixed regression cases passed.
- One-second CPU monitor: 64 concurrent heartbeats, 25.061 ms maximum interval.
- Shipped risk predictor: all five direction invariants passed; the stored temporal validation metrics describe its saved weights, not a newly recreated holdout.
- Assistance co-adaptation: three policy agency audits passed for F1/F3 simulated tuning families, five simulated people per family and two sessions. These are in-sample simulator results; the adaptive policy does not beat the plain interface in every family, and no human benefit or model promotion is claimed.

The [generated proof](../proof.md) includes all simulation policies rather than only the better result. The README uses this fresh snapshot, while the [original release snapshot](evidence/software-benchmarks-2026-10-16.json) remains available. Historical TTS and subscription-planning samples retain their August dates and original scopes.

## Search and agent discovery

The landing repository still advertised v0.13.0, 157 actions, 18 independent verifiers, and August measurements. It now describes published v0.14.0, 162 declared actions, 21 executable specialists, and 22 independent post-condition verifiers. This preserves the distinction between provider coverage and independently observed outcomes.

Updated surfaces include the visible homepage and proof page, Markdown counterparts, identity page, release feeds, capability catalog, source and website LLM indexes, accessibility guide, comparison coverage, FAQ, structured software/dataset metadata, and sitemap dates for changed pages only. The recurring visibility pack has 42 prompts, including co-adaptation evidence and user-configured MCP tools.

The website's `scripts/sync_product_evidence.py` mirrors product evidence and generates the compact documentation-MCP record. Its `--check` mode detects drift. MCP checks compare the evidence source hash, metrics and simulation boundaries with the public `benchmarks.json` mirror. Site metadata validation checks canonical URLs, descriptions, unique titles, headings, JSON-LD and sitemap dates across all 23 sitemap URLs. FAQ schema is restricted to the static identity page, whose questions and answers remain visible; the cinematic homepage makes its legacy main content inert.

Live checks observed an HTTP 308 redirect from the apex domain to `www`, HTTP 200 for the canonical homepage, and crawl access in robots.txt. Before publication, the website and retrieved search result still showed older release facts. After deployment, the homepage, proof page, Markdown negotiation, release/capability JSON, raw benchmark mirror and documentation MCP returned current evidence. Google's [AI search guidance](https://developers.google.com/search/docs/appearance/ai-features) supports the same core practices: crawlable useful content and structured data matching visible text.

Real Chrome sessions tested ChatGPT (unpersonalized temporary chat; UI label Pro) and Gemini (temporary chat; UI label Auto), without supplying answers. ChatGPT discovered v0.14.0 and the October benchmark through GitHub, although its retrieved website excerpt was older. Gemini returned the prior release and August metrics, and confused the short phrase "Heliox assistance co-adaptation" with medical gas. The feature's SEO copy now explicitly identifies Heliox OS desktop software. These failures are preserved, not replaced with source-guided successes. See the [dated public visibility audit](https://www.helioxos.dev/ai-visibility.md) and its raw capture links.

## Verification and publication status

- Product: 12 targeted benchmark, public-proof, capability and release-feed tests passed; changed Python lint and whitespace checks passed.
- Website: production CSS build; 11 content-negotiation tests; 3 public-API tests; visibility, IndexNow and cross-timezone sitemap unit tests; six-tool documentation-MCP contract; all nine website discovery/content validators; product-mirror check.
- Deployment verification found and fixed the raw benchmark route's initial 404. CI found and fixed a date-only sitemap check that depended on the runner's timezone. Regression checks cover both cases.
- Scoped benchmark/SEO commits were pushed to both `main` branches and the website's production deployments succeeded. IndexNow accepted all 23 canonical URLs with HTTP 200; this is notification acceptance, not indexing or ranking proof.
- [Product CI](https://github.com/VyomKulshrestha/Heliox-OS/actions/workflows/ci.yml) and [website CI](https://github.com/VyomKulshrestha/Heliox-OS-LandingPage/actions/workflows/website-ci.yml) remain authoritative for each commit's checks. No ranking increase, traffic increase, completed recrawl, or universal assistant-discovery success is claimed.
