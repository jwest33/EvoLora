# Tech Support Router Evolution Demo

## Overview
This demo showcases the R-Zero inspired co-evolution system by evolving routing instructions for a technical support query classification system. The system demonstrates how a Challenger agent and Solver agent co-evolve to handle increasingly complex routing decisions.

## Quick Start

### Windows
```bash
# Run the full demo (30 generations)
demo\run_tech_support_demo.bat

# Or run a quick test (3 generations)
demo\test_tech_support_demo.bat
```

### Linux/Mac
```bash
# Make executable and run
chmod +x /demo/run_tech_support_demo.sh
./demo/run_tech_support_demo.sh
```

## Demo Configuration

### Hierarchy Structure
The demo uses a realistic tech support hierarchy with 4 main categories:

```
tech_support/
├── account/
│   ├── login_issues       # Password resets, 2FA, lockouts
│   ├── permissions         # Access control, roles
│   └── profile            # Settings, preferences
├── technical/
│   ├── api_errors         # API failures, rate limiting
│   ├── performance        # Slowness, timeouts
│   ├── bugs              # Crashes, unexpected behavior
│   └── integration       # Third-party connections
├── billing/
│   ├── payments          # Failed payments, card issues
│   ├── subscriptions     # Plan changes, cancellations
│   ├── invoices          # Billing history, receipts
│   └── refunds           # Refund requests, disputes
└── product/
    ├── features          # Feature requests, capabilities
    ├── usage            # How-to questions, tutorials
    └── documentation    # Docs requests, clarifications
```

### Evolution Parameters
- **Generations**: 30 (configurable)
- **Population Size**: 15 instruction variants per generation
- **Dataset Size**: 40 queries generated per generation
- **Initial Difficulty**: 0.3 (30% ambiguity)
- **Max Difficulty**: 0.95 (95% ambiguity)

## Expected Evolution Timeline

### Generation 1-5: Bootstrap Phase
- **Accuracy**: 20-35%
- **Difficulty**: 0.3-0.4
- **Instruction Type**: Basic pattern matching
- **Example**: "Route to the category that matches the query"

### Generation 10-15: Learning Phase
- **Accuracy**: 45-60%
- **Difficulty**: 0.5-0.6
- **Instruction Type**: Keyword recognition
- **Example**: "Look for keywords like 'login', 'payment', 'API' to determine the category"

### Generation 20-25: Optimization Phase
- **Accuracy**: 65-75%
- **Difficulty**: 0.7-0.8
- **Instruction Type**: Context understanding
- **Example**: "Identify the primary issue. For login problems check account/login_issues. For payment failures check billing/payments. Prioritize technical issues if error codes are mentioned."

### Generation 30: Convergence
- **Accuracy**: 80-85%
- **Difficulty**: 0.85-0.9
- **Instruction Type**: Sophisticated multi-factor routing
- **Example**: "Analyze the query for primary concern and context. Route authentication and access issues to account, prioritizing login_issues for credential problems. Technical errors with codes or API mentions go to technical branches. Financial matters route to billing. Default to product for usage questions."

## Key Metrics to Monitor

### 1. Co-Evolution Dynamics
Watch how Challenger difficulty increases as Solver accuracy improves:
- Challenger increases difficulty when Solver exceeds 70% accuracy
- Optimal performance zone: 60-70% accuracy (maximum learning signal)

### 2. Instruction Evolution
Track how instructions evolve from simple to complex:
- Length: 20-30 words → 80-100 words
- Specificity: Generic → Domain-specific
- Structure: Single rule → Multi-condition logic

### 3. Failure Pattern Analysis
Monitor common misclassifications:
- Cross-category queries (e.g., "Can't login to view invoices")
- Ambiguous terminology (e.g., "account" could mean user account or billing account)
- Multi-issue queries (e.g., "API is slow and returning errors")

## Output Files

### Main Results
- `evolution_runs/tech_support_demo/best_instruction.txt` - Final evolved instruction
- `evolution_runs/tech_support_demo/evolution_metrics.json` - Performance over time
- `evolution_runs/tech_support_demo/final_dataset.csv` - Generated test queries

### Generation Snapshots
- `evolution_runs/tech_support_demo/generation_N/population.json` - All instructions
- `evolution_runs/tech_support_demo/generation_N/challenger_queries.csv` - Generated queries
- `evolution_runs/tech_support_demo/generation_N/solver_results.json` - Routing performance

## Screenshot Opportunities

### 1. Initial State
Capture the first generation showing random routing with ~25% accuracy.

### 2. Mid-Evolution Progress
Show generation 15 with improving metrics and Challenger adapting difficulty.

### 3. Performance Graph
Plot accuracy vs difficulty over 30 generations showing co-evolution.

### 4. Best Instruction Evolution
Compare instructions from generation 1, 15, and 30 to show sophistication growth.

### 5. Challenging Query Examples
Show examples of difficult queries the Challenger generates:
- "My API key isn't working and I think I'm being overcharged"
- "The dashboard timeout is preventing me from updating billing"
- "Need admin access to fix integration authentication errors"

## Interpreting Results

### Success Indicators
- Final accuracy > 80%
- Smooth co-evolution (no sudden drops)
- Instruction handles edge cases
- Diverse failure patterns (not stuck on one category)

### Common Patterns
- Early generations focus on single keywords
- Mid generations develop category priorities
- Late generations handle ambiguity and context

### Typical Final Instruction
A successful evolution produces instructions that:
1. Identify primary vs secondary issues
2. Prioritize certain categories for ambiguous cases
3. Use technical indicators (error codes, API mentions)
4. Provide fallback routing logic

## Troubleshooting

### Low Final Accuracy (<70%)
- Increase generations to 50
- Adjust mutation rate to 0.3
- Check if hierarchy is too ambiguous

### Evolution Stagnates
- Increase population size to 20
- Adjust difficulty increment to 0.03
- Verify LLM server is responding properly

### Too Fast Convergence
- Increase initial difficulty to 0.5
- Reduce elite size to 2
- Increase dataset size to 60

## Extensions for Documentation

### 1. A/B Testing
Run same configuration twice to show variability in evolution paths.

### 2. Difficulty Modes
- Easy: 20 generations, difficulty cap at 0.7
- Normal: 30 generations, difficulty cap at 0.9
- Hard: 50 generations, no difficulty cap

### 3. Hierarchy Complexity
- Simple: 2 levels, 4 leaf nodes
- Medium: 3 levels, 12 leaf nodes (default)
- Complex: 4 levels, 20+ leaf nodes

## Expected Demo Duration

- **Quick Test**: 1-2 minutes (3 generations)
- **Standard Demo**: 5-10 minutes (30 generations)
- **Full Evolution**: 15-20 minutes (50 generations)
- **Extended Run**: 30-45 minutes (100 generations)

## Key Takeaways for Users

1. **Self-Improving System**: No manual labeling required
2. **Adversarial Training**: Challenger keeps Solver learning
3. **Domain Agnostic**: Works with any hierarchical classification
4. **Progressive Curriculum**: Difficulty adapts to performance
5. **Interpretable Evolution**: Can trace instruction improvements
