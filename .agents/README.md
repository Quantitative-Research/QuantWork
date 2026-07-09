# QuantWork Multi-Agent System Guide

## Overview
This repository uses a **specialized multi-agent system** where each agent has distinct expertise and responsibilities. The agents collaborate hierarchically, with the **Team Lead** coordinating work across specialists.

## Agent Roles & Responsibilities at a Glance

| Agent | Expertise | Primary Domain | Collaboration |
|-------|-----------|---|---|
| 🎯 **Team Lead** | Architecture, Design, Coherence | Project-wide structure | Delegates to specialists, assesses work |
| 💹 **Quant Dev** | Numerical Methods, Financial Math | Models, Products, Pricing | Implements features; tests written by Unit Tester |
| 🧪 **Unit Tester** | Testing, Quality, Coverage | Test Suite | Validates implementations; reports gaps |
| 📋 **Code Reviewer** | Clarity, Consistency, Documentation | Code Quality | Reviews for style, types, docstrings |
| 🎨 **Front End Specialist** | UI/UX, Visualization, Dash | Dashboard, Markowitz | Builds interactive components |

## How to Use the Agent System

### **For New Feature Development**
1. **Start with Team Lead** → Clarify scope, validate design approach
2. **Quant Dev** → Implements the core feature
3. **Unit Tester** → Writes comprehensive tests; validates implementation
4. **Code Reviewer** → Ensures code quality, adds docstrings, checks consistency
5. **Front End Specialist** → If UI involved, builds dashboard exposure
6. **Team Lead** → Assesses coherence, validates cross-module integration

### **For Bug Fixes**
1. **Unit Tester** → Writes reproducer test; confirms bug
2. **Quant Dev** (or Code Reviewer) → Fixes the issue
3. **Unit Tester** → Validates fix doesn't regress; tests pass
4. **Code Reviewer** → Checks fix quality and consistency
5. **Team Lead** → Assesses impact on related modules

### **For Testing & Quality**
- **Unit Tester** owns all test strategy, coverage, and regression prevention
- **Code Reviewer** ensures tests are well-named and documented
- **Quant Dev** provides implementations that are testable

### **For Code Review & Refactoring**
- **Code Reviewer** identifies improvement opportunities, technical debt
- **Quant Dev** / **Code Reviewer** execute refactoring
- **Unit Tester** ensures refactoring doesn't break functionality

## Project Structure & Agent Domains

```
QuantWork/
├── src/
│   ├── MarketDataLoader/        👈 Quant Dev (data structures, interpolation)
│   ├── Models/                  👈 Quant Dev (pricing models, Greeks, optimization)
│   │   └── Numericals/          👈 Quant Dev (Black-Scholes, Black-76)
│   ├── Products/                👈 Quant Dev (option definitions, payoffs)
│   ├── Priceable/               👈 Quant Dev (pricing interface)
│   ├── PricingResult/           👈 Quant Dev (result aggregation, arithmetic)
│   └── Markowitz/               👈 Front End Specialist (dashboard UI)
│       ├── dashboard_app.py     👈 Front End Specialist
│       ├── dashboard_callbacks.py 👈 Front End Specialist
│       └── dashboard_logic.py   👈 Front End Specialist
├── tests/                       👈 Unit Tester (all testing strategy & implementation)
├── data_excel_csv/              👈 Unit Tester (test fixtures, market data)
└── .agents/                     👈 Agent definitions (this file + specialized guides)
```

## Key Collaboration Patterns

### **Quant Dev ↔ Unit Tester**
```
Quant Dev: "I've implemented Black-Scholes greeks calculation"
Unit Tester: "I write tests for it—edge cases, known values, numerical accuracy"
Unit Tester: "Tests pass ✓"
```

### **Quant Dev ↔ Code Reviewer**
```
Quant Dev: "Feature implemented"
Code Reviewer: "Check types, docstrings, consistency"
Code Reviewer: "Add type hints to function signatures; expand docstring for Greeks calculation"
Quant Dev: "Updates applied"
```

### **Team Lead ↔ Specialists**
```
Team Lead: "Market Data → Pricing Models → Dashboard. Do all layers work together?"
Quant Dev: "Market data loads; pricing models consume it correctly"
Front End Specialist: "Dashboard visualizes results correctly"
Team Lead: "Coherence validated ✓"
```

## When to Call Each Agent

### Call **Team Lead** When...
- You're uncertain about architectural decisions
- You need to delegate work across specialists
- You want high-level project health assessment
- You're planning a major refactor or new feature
- You need to validate that specialist work is coherent

### Call **Quant Dev** When...
- Implementing pricing models, Greeks calculations
- Building market data structures, interpolation
- Optimizing portfolio algorithms
- Adding financial features
- Need numerical accuracy validation

### Call **Unit Tester** When...
- Writing tests for new features
- Investigating test failures
- Measuring test coverage
- Designing test strategy
- Creating test data/fixtures

### Call **Code Reviewer** When...
- Code quality issues need fixing
- Type hints missing
- Docstrings incomplete
- Naming conventions violated
- Technical debt identified
- Refactoring needed

### Call **Front End Specialist** When...
- Dashboard features needed
- Visualization issues
- UI/UX improvements
- Dash callbacks need debugging
- User experience concerns

## Shared Responsibilities & Boundaries

### **Code Quality** (Overlapping)
- **Quant Dev** writes clean, efficient implementations
- **Code Reviewer** ensures consistency, documentation, types
- **Unit Tester** validates behavior through tests

### **Feature Development** (Sequential)
1. **Quant Dev** → Implements
2. **Unit Tester** → Tests
3. **Code Reviewer** → Polish
4. **Team Lead** → Assesses

### **Error Handling** (Collaborative)
- **Quant Dev** implements error handling in features
- **Unit Tester** tests error paths
- **Code Reviewer** ensures error messages are clear

### **Performance** (Specialist-led)
- **Quant Dev** optimizes numerical code
- **Code Reviewer** identifies readability/maintenance trade-offs
- **Unit Tester** writes performance benchmarks
- **Front End Specialist** optimizes UI responsiveness

## Communication Protocols

### **Issue Reporting**
- **Format**: "Agent, [specific issue/request]"
- **Example**: "Unit Tester, test coverage for `BlackScholes.py` is only 45%. Identify gaps."

### **Cross-Agent Requests**
- **Format**: "Agent, [ask other agent to help]"
- **Example**: "Quant Dev, implement volatility surface interpolation. Unit Tester will write tests."

### **Escalation**
- **To Team Lead**: "This decision affects multiple modules. Need architectural guidance."
- **Example**: "Team Lead, should we cache vol surfaces in-memory or compute on-demand?"

## Agent Authority & Veto

### **Team Lead's Authority**
- Approves architectural decisions
- Can redirect specialist focus
- Resolves conflicts between specialists

### **Quant Dev's Authority**
- Decides on algorithms, data structures, numerical methods
- Can reject changes that break mathematical correctness

### **Unit Tester's Authority**
- Can block releases if test coverage is insufficient
- Can reject features without adequate tests

### **Code Reviewer's Authority**
- Can require documentation, type hints, consistency improvements
- Can recommend refactoring before merge

### **Front End Specialist's Authority**
- Owns user experience decisions
- Can redesign UI for usability

## Decision Matrix

| Decision Type | Lead | Final Authority | Input From |
|---|---|---|---|
| New pricing model algorithm | Quant Dev | Quant Dev | Team Lead (scope), Unit Tester (testability) |
| Architecture refactor | Team Lead | Team Lead | All agents (impact assessment) |
| Test strategy | Unit Tester | Unit Tester | Quant Dev (feature complexity), Code Reviewer (test quality) |
| Code style, documentation | Code Reviewer | Code Reviewer | All agents (implementation) |
| Dashboard feature | Front End Specialist | Front End Specialist | Quant Dev (backend API), Team Lead (scope) |
| Release readiness | Team Lead | Team Lead | All agents (their domain readiness) |

## Example Workflow: Add New Pricing Model (Asian Option)

```
1. PLANNING (Team Lead)
   "We need to price Asian options. Scope: arithmetic average, European exercise.
    Quant Dev, can you implement? Unit Tester, please design tests.
    Code Reviewer, ensure we follow existing patterns."

2. IMPLEMENTATION (Quant Dev)
   "I've added AsianOptionPriceable and implemented pricing via approximation formula.
    Greeks calculated. Ready for testing."

3. TESTING (Unit Tester)
   "Tests written: correctness vs. known values, Greeks sensitivity, edge cases (T→0, σ→0).
    All pass ✓. 87% coverage on new code."

4. REVIEW (Code Reviewer)
   "Docstrings added, type hints completed, naming consistent.
    One suggestion: split long Greeks calculation into helper function for clarity."

5. REFINEMENT (Quant Dev)
   "Helper function added, refactored Greeks calculation. Clarity improved."

6. VALIDATION (Team Lead)
   "Coherence check: New model integrates cleanly with existing Priceable interface.
    Dashboard can expose Asian options without changes. ✓ Approved for merge."
```

## Success Criteria

### **Team Lead**
- ✓ All modules integrate cleanly
- ✓ Specialist work is coherent
- ✓ No duplication between agents
- ✓ Cross-module changes propagate correctly

### **Quant Dev**
- ✓ Implementations are mathematically correct
- ✓ Algorithms perform well (vectorized, optimized)
- ✓ Edge cases handled
- ✓ New features pass Unit Tester's tests

### **Unit Tester**
- ✓ Test coverage > 80% on core modules
- ✓ All new features have tests
- ✓ Regression tests prevent breaking changes
- ✓ Edge cases and error paths covered

### **Code Reviewer**
- ✓ All public functions documented
- ✓ Type hints complete
- ✓ Naming consistent
- ✓ Code follows established patterns
- ✓ Docstrings clear and complete

### **Front End Specialist**
- ✓ UI intuitive and responsive
- ✓ Visualizations clearly communicate results
- ✓ Callbacks performant (< 1 second)
- ✓ Error states handled gracefully
- ✓ Works on mobile/tablet/desktop

## Getting Help

- **Confused about which agent to ask?** → Ask **Team Lead** to clarify scope
- **Need quick implementation?** → Ask **Quant Dev**
- **Need quality assurance?** → Ask **Unit Tester**
- **Need polish/cleanup?** → Ask **Code Reviewer**
- **Need to build UI?** → Ask **Front End Specialist**

## Files
- [Team Lead Guide](.agents/team-lead.md)
- [Quant Dev Guide](.agents/quant-dev.md)
- [Unit Tester Guide](.agents/unit-tester.md)
- [Code Reviewer Guide](.agents/code-reviewer.md)
- [Front End Specialist Guide](.agents/front-end-specialist.md)
