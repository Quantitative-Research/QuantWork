# Quick Reference: Invoke Your Agents

This file shows how to invoke each specialized agent in the QuantWork project.

## 📋 Agent Reference Card

### 🎯 Team Lead
**Expert in**: Architecture, design coherence, delegation, high-level assessment

**Ask for help with**:
- "What's the best architectural approach for [feature]?"
- "Should we refactor the market data layer?"
- "Does this code change affect other modules?"
- "Assess coherence of specialist work on [task]"

**What they can't help with**: Writing code, implementing features, UI design, math details

**File**: `.agents/team-lead.md`

---

### 💹 Quant Dev  
**Expert in**: Pricing models, numerical methods, Greeks, algorithms, market data structures

**Ask for help with**:
- "Implement [pricing model] with full Greeks"
- "Debug volatility surface interpolation"
- "Optimize portfolio optimization algorithm"
- "Calculate Greeks for this exotic payoff"
- "Design market data structure for [feature]"

**What they can't help with**: Testing, documentation, UI, code style, architecture decisions

**File**: `.agents/quant-dev.md`

---

### 🧪 Unit Tester
**Expert in**: Test design, coverage, regression prevention, fixtures, test data

**Ask for help with**:
- "Design tests for [feature]"
- "Why is test [name] failing?"
- "What's test coverage for [module]?"
- "Write edge case tests for [function]"
- "Create test fixtures for market data scenario"

**What they can't help with**: Fixing bugs in production code, design decisions, UI, refactoring

**File**: `.agents/unit-tester.md`

---

### 📋 Code Reviewer
**Expert in**: Documentation, type hints, consistency, API design, code clarity

**Ask for help with**:
- "Add docstrings and type hints to [file]"
- "Why is this code unclear?"
- "Should we refactor [function] for clarity?"
- "Is this API design consistent with the rest?"
- "Review [module] for code quality"

**What they can't help with**: Fixing bugs, implementing features, testing strategy, performance optimization

**File**: `.agents/code-reviewer.md`

---

### 🎨 Front End Specialist
**Expert in**: Dashboard UI, Dash callbacks, visualization, user experience

**Ask for help with**:
- "Build a [component] for the dashboard"
- "Why is the dashboard slow?"
- "Add visualization for [data]"
- "Design UI for asset picker"
- "Debug Dash callback for [interaction]"

**What they can't help with**: Pricing models, testing backend, code consistency, architecture

**File**: `.agents/front-end-specialist.md`

---

## 📂 Using Agents in Your Workflow

### Scenario 1: Implement a New Feature
1. **Team Lead**: "Is this a good fit architecturally? What should we delegate?"
2. **Quant Dev**: "Implement the core feature"
3. **Unit Tester**: "Write tests, validate coverage"
4. **Code Reviewer**: "Review code quality, add docstrings"
5. **Team Lead**: "Assess coherence across modules"

### Scenario 2: Debug a Test Failure
1. **Unit Tester**: "Why is this test failing?"
2. **Quant Dev** OR **Code Reviewer**: Fix the issue (depends on root cause)
3. **Unit Tester**: "Validate fix works; check for regressions"

### Scenario 3: Refactor Code for Clarity
1. **Code Reviewer**: "Identify improvement opportunities"
2. **Quant Dev** OR **Code Reviewer**: Execute refactoring
3. **Unit Tester**: "Ensure no regression"
4. **Team Lead**: "Validate architectural coherence"

### Scenario 4: Improve Dashboard
1. **Front End Specialist**: "Design and implement UI improvements"
2. **Code Reviewer**: "Review code quality"
3. **Unit Tester**: "Test callbacks and interactions"

---

## 🎯 Recommended Agent Assignment by Code Area

| Code Area | Primary Agent | Secondary Agents |
|-----------|---|---|
| `src/Models/Numericals/` | **Quant Dev** | Unit Tester, Code Reviewer |
| `src/Models/BlackModel.py` | **Quant Dev** | Unit Tester, Code Reviewer |
| `src/MarketDataLoader/` | **Quant Dev** | Code Reviewer, Unit Tester |
| `src/Products/` | **Quant Dev** | Unit Tester, Code Reviewer |
| `src/Priceable/` | **Quant Dev** | Unit Tester, Code Reviewer |
| `src/PricingResult/` | **Quant Dev** | Unit Tester, Code Reviewer |
| `src/Markowitz/dashboard_*.py` | **Front End Specialist** | Unit Tester, Code Reviewer |
| `tests/` | **Unit Tester** | Code Reviewer, Quant Dev |
| Project Architecture | **Team Lead** | All others (input) |
| Code Quality, Types, Docs | **Code Reviewer** | Quant Dev, Unit Tester |

---

## 💡 Quick Decision: Which Agent?

**Need to implement a feature?** → **Quant Dev** (if models/data) or **Front End Specialist** (if UI)

**Need to test it?** → **Unit Tester**

**Need to make it look good?** → **Code Reviewer**

**Need architectural advice?** → **Team Lead**

**Need UI improvements?** → **Front End Specialist**

**Unsure who to ask?** → **Team Lead** (they'll delegate)

---

## 🔗 See Also
- [Main Agent Guide](.agents/README.md)
- [Team Lead Details](.agents/team-lead.md)
- [Quant Dev Details](.agents/quant-dev.md)
- [Unit Tester Details](.agents/unit-tester.md)
- [Code Reviewer Details](.agents/code-reviewer.md)
- [Front End Specialist Details](.agents/front-end-specialist.md)
