#!/bin/bash

echo "========================================"
echo "Tech Support Router Evolution Demo"
echo "========================================"
echo ""
echo "This demo evolves routing instructions for a technical support"
echo "query classification system using R-Zero inspired co-evolution."
echo ""
echo "The Challenger agent will generate increasingly difficult queries"
echo "while the Solver agent evolves better routing instructions."
echo ""
echo "Configuration:"
echo "  - Hierarchy: Tech support with 4 main categories"
echo "  - Categories: Account, Technical, Billing, Product"
echo "  - Generations: 30 (for demo, ~5-10 minutes)"
echo "  - Population: 15 instruction variants per generation"
echo "  - Dataset: 40 queries generated per generation"
echo ""
echo "Starting evolution..."
echo "----------------------------------------"

# Create output directory
mkdir -p evolution_runs/tech_support_demo

# Run the evolution with moderate settings for demo
python -m embedlab.cli_evolution evolve \
  --hierarchy data/tech_support_hierarchy.yaml \
  --output evolution_runs/tech_support_demo \
  --generations 30 \
  --population 15 \
  --verbose

echo ""
echo "========================================"
echo "Evolution Complete!"
echo "========================================"
echo "Results saved to: evolution_runs/tech_support_demo/"
echo ""
echo "Check the following files:"
echo "  - best_instruction.txt: The evolved routing instruction"
echo "  - evolution_metrics.json: Performance metrics over time"
echo "  - generation_*/: Detailed logs for each generation"
echo ""
