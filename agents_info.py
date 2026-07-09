#!/usr/bin/env python3
"""
QuantWork Multi-Agent System Setup Checklist

Use this checklist to verify your agent system is ready to use.
"""

AGENTS = {
    "Team Lead": {
        "file": ".agents/team-lead.md",
        "domain": "Architecture, design coherence, delegation",
        "primary_modules": [
            "Project-wide structure",
            "Integration points between modules",
            "High-level design decisions"
        ],
        "skills": [
            "High-level architecture",
            "Design pattern validation",
            "Cross-module coherence",
            "Delegation strategy",
            "Technical debt assessment"
        ]
    },
    "Quant Dev": {
        "file": ".agents/quant-dev.md",
        "domain": "Numerical methods, pricing models, Greeks, algorithms",
        "primary_modules": [
            "src/Models/",
            "src/Models/Numericals/",
            "src/MarketDataLoader/",
            "src/Products/",
            "src/Priceable/",
            "src/PricingResult/"
        ],
        "skills": [
            "Black-Scholes pricing",
            "Black-76 pricing",
            "Greeks calculations",
            "Volatility surface interpolation",
            "Forward curve construction",
            "Portfolio optimization",
            "Numerical stability",
            "Algorithm optimization",
            "Edge case handling"
        ]
    },
    "Unit Tester": {
        "file": ".agents/unit-tester.md",
        "domain": "Testing, coverage, regression prevention, test design",
        "primary_modules": [
            "tests/",
            "data_excel_csv/",
            "Test infrastructure"
        ],
        "skills": [
            "Unit test design",
            "Integration test design",
            "End-to-end test design",
            "Mocking external dependencies",
            "Test data/fixture creation",
            "Coverage analysis",
            "Edge case identification",
            "Regression test design",
            "Performance benchmarking"
        ]
    },
    "Code Reviewer": {
        "file": ".agents/code-reviewer.md",
        "domain": "Code quality, consistency, documentation, clarity",
        "primary_modules": [
            "All src/ modules (quality focus)",
            "All tests/ (test code quality)",
            "Documentation"
        ],
        "skills": [
            "Python style (PEP 8)",
            "Type hints (PEP 484)",
            "Docstring writing",
            "Naming conventions",
            "API design",
            "Code clarity",
            "Technical debt identification",
            "Refactoring recommendations",
            "Documentation quality"
        ]
    },
    "Front End Specialist": {
        "file": ".agents/front-end-specialist.md",
        "domain": "Dashboard UI, visualization, user experience, Dash",
        "primary_modules": [
            "src/Markowitz/dashboard_app.py",
            "src/Markowitz/dashboard_callbacks.py",
            "src/Markowitz/dashboard_logic.py"
        ],
        "skills": [
            "Dash framework",
            "Plotly visualization",
            "React principles",
            "Callback design",
            "State management",
            "Bootstrap Grid",
            "Responsiveness",
            "UX/UI principles",
            "Data visualization",
            "Performance optimization"
        ]
    }
}

def print_setup_checklist():
    """Print setup verification checklist."""
    print("\n" + "="*70)
    print("QuantWork Multi-Agent System Setup Checklist")
    print("="*70 + "\n")
    
    for agent_name, agent_info in AGENTS.items():
        print(f"✓ {agent_name}")
        print(f"  File: {agent_info['file']}")
        print(f"  Domain: {agent_info['domain']}")
        print(f"  Primary modules: {', '.join(agent_info['primary_modules'][:2])}{'...' if len(agent_info['primary_modules']) > 2 else ''}")
        print()

def print_delegation_guide():
    """Print quick delegation guide."""
    print("\n" + "="*70)
    print("Quick Delegation Guide")
    print("="*70 + "\n")
    
    delegations = [
        ("Implement pricing model", "Quant Dev"),
        ("Write tests for feature", "Unit Tester"),
        ("Add documentation", "Code Reviewer"),
        ("Design dashboard", "Front End Specialist"),
        ("Assess architecture", "Team Lead"),
        ("Debug numerical issue", "Quant Dev"),
        ("Improve code clarity", "Code Reviewer"),
        ("Expand test coverage", "Unit Tester"),
        ("Validate cross-module coherence", "Team Lead"),
    ]
    
    for task, agent in delegations:
        print(f"{task:.<40} → {agent}")

def print_workflow_example():
    """Print example multi-agent workflow."""
    print("\n" + "="*70)
    print("Example Workflow: Add New Pricing Model (European Barrier Option)")
    print("="*70 + "\n")
    
    workflow = [
        ("1. Team Lead", "Review scope. Validate it aligns with architecture. Delegate."),
        ("2. Quant Dev", "Implement barrier option pricing with Greeks. Pure numerical code."),
        ("3. Unit Tester", "Write tests: correctness, edge cases, Greeks sensitivity. Validate coverage."),
        ("4. Code Reviewer", "Review for types, docstrings, consistency. Refactor for clarity if needed."),
        ("5. Front End Specialist", "If needed, add dashboard visualization of barrier option results."),
        ("6. Team Lead", "Assess integration. Validate coherence. Approve for merge."),
    ]
    
    for step, description in workflow:
        print(f"{step:.<20} {description}")

if __name__ == "__main__":
    print_setup_checklist()
    print_delegation_guide()
    print_workflow_example()
    
    print("\n" + "="*70)
    print("Agent System Ready!")
    print("="*70)
    print("\nSee AGENTS.md for quick reference or .agents/README.md for detailed guide.")
    print("\n")
