# Copilot Instructions for QuantWork

This repository is a quantitative finance library with a clear separation between:
- market data loading
- pricing models and Greeks
- option/product definitions
- portfolio optimization
- the Dash-based dashboard

Use the most appropriate specialized chat mode for the task:
- Team Lead for architecture, delegation, and cross-module coherence
- Quant Dev for pricing models, numerical logic, and market-data processing
- Unit Tester for tests, regressions, and edge cases
- Code Reviewer for clarity, documentation, typing, and API consistency
- Front End Specialist for dashboard UI and visual results

When helping in this repository:
- prefer concise, repository-aware answers
- keep changes scoped to the relevant module
- preserve the existing library structure and abstractions
- validate behavior with tests when changing core logic
- avoid mixing UI work with pricing-model work unless the request explicitly spans both
