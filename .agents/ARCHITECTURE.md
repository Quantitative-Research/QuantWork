# QuantWork Agent Architecture

## Visual Overview

```
                         ┌─────────────────┐
                         │   TEAM LEAD     │
                         │  🎯 Architect   │
                         │  Coordinator    │
                         │  Assessor       │
                         └────────┬────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  │               │               │
        ┌─────────▼────────┐   ┌──▼──────────┐   │
        │   QUANT DEV      │   │ UNIT TESTER │   │
        │ 💹 Senior Dev    │   │ 🧪 QA Lead  │   │
        │ Models & Pricing │   │ Testing     │   │
        └──────────────────┘   └─────────────┘   │
                                                  │
                    ┌─────────────────────────────┤
                    │                             │
        ┌───────────▼────────────┐   ┌───────────▼──────────┐
        │   CODE REVIEWER        │   │ FRONT END SPECIALIST │
        │ 📋 Quality Guardian    │   │ 🎨 UI/UX Expert     │
        │ Clarity & Consistency  │   │ Dashboard Builder    │
        └────────────────────────┘   └──────────────────────┘
```

## Domain Mapping

```
                    QuantWork Project Structure
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    ┌───▼────┐         ┌───▼────┐         ┌──▼───┐
    │  DATA  │         │ MODELS │         │ DASH │
    │LOADING │         │ PRICING │        │ UI   │
    └────────┘         └────────┘         └──────┘
        │                  │                  │
        │                  │                  │
    💹 QUANT DEV       💹 QUANT DEV     🎨 FRONT END
    (structures,       (algorithms,       (callbacks,
     interpolation)     Greeks)           visualization)
        │                  │                  │
    🧪 UNIT TESTER     🧪 UNIT TESTER    🧪 UNIT TESTER
    (fixtures,         (correctness,     (interaction
     loading tests)     edge cases)        tests)
        │                  │                  │
    📋 CODE REVIEWER   📋 CODE REVIEWER   📋 CODE REVIEWER
    (clarity,          (documentation,   (clarity,
     consistency)      types)             style)
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    🎯 TEAM LEAD
                    (coherence check)
```

## Feature Development Pipeline

```
IDEA
 │
 ├─→ 🎯 Team Lead (Scope & Approach)
 │       "Is this architecturally sound?"
 │       ▼
 ├─→ 💹 Quant Dev (Implementation)
 │       "Implement feature with quality code"
 │       ▼
 ├─→ 🧪 Unit Tester (Validation)
 │       "Write tests; validate coverage"
 │       ▼
 ├─→ 📋 Code Reviewer (Polish)
 │       "Add docs, types, consistency"
 │       ▼
 ├─→ 🎨 Front End (UI Exposure)
 │       [if applicable] "Build dashboard"
 │       ▼
 ├─→ 🎯 Team Lead (Coherence)
 │       "Validate cross-module integration"
 │       ▼
 └─→ SHIPPED ✅
```

## Responsibility Boundaries

```
Component          | Quant Dev | Tester | Reviewer | FE | Team Lead
─────────────────────────────────────────────────────────────────────
Model Implementation    | ★★★     |        |    ★    |   |   ★
Numerical Accuracy      | ★★★     | ★★★    |         |   |   
Code Quality            | ★       | ★      |  ★★★    |   |
Documentation           | ★       |        |  ★★★    |   |
Testing Strategy        |         | ★★★    |    ★    |   |   ★
Test Coverage           |         | ★★★    |         |   |
Dashboard UI            |         |    ★   |    ★    |★★★|   ★
Performance             | ★★      | ★      |    ★    |★★ |
Architecture            |    ★    |        |    ★    |   |  ★★★

★★★ = Primary Responsibility
★★  = Secondary Responsibility  
★   = Contributing/Advisory Role
```

## Communication Channels

```
Daily Collaboration:
  Quant Dev ←→ Unit Tester (implement + test loops)
  Code Reviewer ←→ Everyone (continuous feedback)

Weekly Integration:
  Team Lead ↔ All Agents (coherence checks, blockers)

Feature Handoffs:
  Quant Dev → Unit Tester → Code Reviewer → [FE if needed] → Team Lead
```

## Decision Authority

```
Architectural Decisions  ────→  🎯 Team Lead (final say)
                                 Input: Quant Dev, Code Reviewer

Algorithm/Model Choices  ────→  💹 Quant Dev (final say)
                                 Input: Team Lead (scope), Unit Tester (testability)

Test Strategy            ────→  🧪 Unit Tester (final say)
                                 Input: Quant Dev (complexity), Code Reviewer (test quality)

Code Quality Standards   ────→  📋 Code Reviewer (final say)
                                 Input: All agents (implementation)

Dashboard/UI Design      ────→  🎨 Front End Specialist (final say)
                                 Input: Team Lead (scope), Quant Dev (API design)

Release Readiness        ────→  🎯 Team Lead (final say)
                                 Input: All agents (their domain readiness)
```

## Success Metrics

```
🎯 Team Lead
   • Architecture coherence score: 100%
   • Cross-module integration issues: 0
   • Specialist work coordination: smooth

💹 Quant Dev
   • Numerical accuracy: ±1e-10 (appropriate tolerance)
   • Edge cases handled: 100%
   • Test pass rate: 100%
   • Performance: vectorized, optimized

🧪 Unit Tester
   • Coverage: >80% on core modules
   • Regression: 0 unexpected failures
   • Test execution time: <5 min (full suite)
   • Edge case coverage: comprehensive

📋 Code Reviewer
   • Type hint coverage: 100%
   • Documentation coverage: 100%
   • Style violations: 0
   • Code clarity: excellent (peer feedback)

🎨 Front End Specialist
   • UI responsiveness: <1s (all interactions)
   • User satisfaction: high (feedback)
   • Mobile coverage: 100%
   • Accessibility: WCAG AA compliant
```

## When Agents Escalate

```
Quant Dev: "Can't implement this without changing the architecture"
           → Escalate to Team Lead

Unit Tester: "Test coverage gap too large; need to reduce scope or increase effort"
             → Escalate to Team Lead

Code Reviewer: "Code quality requires major refactoring; schedule conflict?"
               → Escalate to Team Lead

Front End: "Dashboard performance hit; need backend optimization"
           → Escalate to Quant Dev + Team Lead

Team Lead: "Can't achieve coherence without resolving conflict between agents"
           → Document decision rationale; proceed
```

## Integration Points

```
Market Data Layer (MarketDataLoader/)
    ↓ [Quant Dev builds]
    ↓ [Unit Tester validates]
    ↓ [Code Reviewer polishes]
    │
    └──→ Data Structures (Market, VolSurface, ForwardCurve)

Pricing Models (Models/)
    ↑ [Consumes: Market data]
    ↓ [Quant Dev builds]
    ↓ [Unit Tester validates]
    ↓ [Code Reviewer polishes]
    │
    └──→ PricingResult (Price + Greeks)

Products (Products/)
    ↑ [Consumes: BlackModel, Market]
    ↓ [Quant Dev builds]
    ↓ [Unit Tester validates]
    ↓ [Code Reviewer polishes]
    │
    └──→ Payoff Definitions

Priceable Interface (Priceable/)
    ↑ [Consumes: PricingModel, Product]
    ↓ [Quant Dev builds]
    ↓ [Unit Tester validates]
    ↓ [Code Reviewer polishes]
    │
    └──→ Pricing Methods

Dashboard (Markowitz/)
    ↑ [Consumes: Optimization API, Priceable interface]
    ↓ [Front End Specialist builds]
    ↓ [Unit Tester validates]
    ↓ [Code Reviewer polishes]
    │
    └──→ User-Facing Results

Team Lead validates: All layers work together ✓
```
