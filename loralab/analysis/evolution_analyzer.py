"""Evolution history analyzer and visualizer

Analyzes the complete evolution history to show:
- Family tree of variants (parent-child relationships)
- Survival patterns and mutation effectiveness
- Performance trends across generations
- Hyperparameter effectiveness
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import pandas as pd
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)


class EvolutionAnalyzer:
    """Analyzes and visualizes evolution history"""

    def __init__(self, output_dir: str):
        """Initialize analyzer with evolution output directory

        Args:
            output_dir: Path to run directory (e.g., lora_runs/run_20250118_093000)
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.history_file = self.output_dir / "history" / "evolution_history.json"
        self.analysis_dir = self.output_dir / "analysis" / "visualizations"
        self.analysis_dir.mkdir(exist_ok=True, parents=True)

        self.history = None
        self.variants = {}  # variant_id -> variant data
        self.lineage = {}  # variant_id -> parent_id
        self.survivors_by_gen = {}  # generation -> list of survivor ids

    def load_history(self) -> bool:
        """Load evolution history from file

        Returns:
            True if history loaded successfully
        """
        if not self.history_file.exists():
            logger.error(f"History file not found: {self.history_file}")
            return False

        try:
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
            logger.info(f"Loaded history with {len(self.history)} generations")
            return True
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return False

    def analyze(self):
        """Run complete analysis of evolution history"""
        if not self.load_history():
            return

        # Build variant database
        self._build_variant_database()

        # Analyze lineage
        self._analyze_lineage()

        # Generate visualizations
        self._create_family_tree()
        self._create_performance_timeline()
        self._create_hyperparameter_heatmap()
        self._create_survival_analysis()
        self._create_mutation_effectiveness()

        # Generate report
        self._generate_report()

        logger.info(f"Analysis complete. Results saved to {self.analysis_dir}")

    def _build_variant_database(self):
        """Build database of all variants across generations"""
        for gen_data in self.history:
            generation = gen_data['generation']

            # Track survivors for this generation
            survivors = []

            for variant in gen_data['variants']:
                variant_id = variant['variant_id']
                self.variants[variant_id] = {
                    'generation': generation,
                    'rank': variant['rank'],
                    'learning_rate': variant['learning_rate'],
                    'dropout': variant['dropout'],
                    'alpha': variant['alpha'],
                    'accuracy': variant['eval_accuracy'],
                    'perplexity': variant['eval_perplexity'],
                    'fitness': variant['eval_accuracy'] - (variant['eval_perplexity'] / 100.0),
                    'parent_id': variant.get('parent_id'),
                    'survived': False  # Will update based on next generation
                }

                # Track lineage
                if variant.get('parent_id'):
                    self.lineage[variant_id] = variant['parent_id']

        # Mark survivors based on presence in next generation
        self._identify_survivors()

    def _identify_survivors(self):
        """Identify which variants survived to next generation"""
        for i in range(len(self.history) - 1):
            current_gen = self.history[i]
            next_gen = self.history[i + 1]

            # Get variant IDs from next generation that are elites
            next_gen_ids = [v['variant_id'] for v in next_gen['variants']]

            # Sort current generation by fitness
            sorted_variants = sorted(
                current_gen['variants'],
                key=lambda v: v['eval_accuracy'] - (v['eval_perplexity'] / 100.0),
                reverse=True
            )

            # Top 2 are survivors (based on keep_top=2)
            survivors = sorted_variants[:2]
            gen_survivors = []

            for survivor in survivors:
                variant_id = survivor['variant_id']
                if variant_id in self.variants:
                    self.variants[variant_id]['survived'] = True
                    gen_survivors.append(variant_id)

            self.survivors_by_gen[current_gen['generation']] = gen_survivors

    def _analyze_lineage(self):
        """Analyze parent-child relationships and mutation patterns"""
        self.mutation_success = {
            'rank': {'total': 0, 'survived': 0},
            'learning_rate': {'total': 0, 'survived': 0},
            'dropout': {'total': 0, 'survived': 0}
        }

        for variant_id, variant_data in self.variants.items():
            parent_id = variant_data.get('parent_id')
            if parent_id and parent_id in self.variants:
                parent = self.variants[parent_id]

                # Check what mutated
                if variant_data['rank'] != parent['rank']:
                    self.mutation_success['rank']['total'] += 1
                    if variant_data['survived']:
                        self.mutation_success['rank']['survived'] += 1

                if variant_data['learning_rate'] != parent['learning_rate']:
                    self.mutation_success['learning_rate']['total'] += 1
                    if variant_data['survived']:
                        self.mutation_success['learning_rate']['survived'] += 1

                if variant_data['dropout'] != parent['dropout']:
                    self.mutation_success['dropout']['total'] += 1
                    if variant_data['survived']:
                        self.mutation_success['dropout']['survived'] += 1

    def _create_family_tree(self):
        """Create family tree visualization showing lineage"""
        fig, ax = plt.subplots(figsize=(20, 12))

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        for variant_id, variant_data in self.variants.items():
            G.add_node(variant_id, **variant_data)

        # Add edges (parent -> child)
        for child_id, parent_id in self.lineage.items():
            if parent_id in self.variants:
                G.add_edge(parent_id, child_id)

        # Position nodes by generation
        pos = {}
        for generation in range(len(self.history)):
            gen_variants = [v for v, d in self.variants.items()
                          if d['generation'] == generation]

            for i, variant_id in enumerate(gen_variants):
                x = generation * 2
                y = i * 1.5 - (len(gen_variants) - 1) * 0.75
                pos[variant_id] = (x, y)

        # Draw nodes with color based on survival
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            variant = self.variants[node]
            if variant['survived']:
                node_colors.append('#2ecc71')  # Green for survivors
                node_sizes.append(500)
            else:
                node_colors.append('#e74c3c')  # Red for eliminated
                node_sizes.append(300)

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                              arrows=True, alpha=0.5, ax=ax)

        # Add labels with shortened IDs
        labels = {}
        for node in G.nodes():
            variant = self.variants[node]
            labels[node] = f"r{variant['rank']}\n{variant['accuracy']:.1%}"

        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

        # Add generation labels
        for gen in range(len(self.history)):
            ax.text(gen * 2, -10, f"Gen {gen}", fontsize=10,
                   ha='center', weight='bold')

        # Legend
        survived_patch = mpatches.Patch(color='#2ecc71', label='Survived')
        eliminated_patch = mpatches.Patch(color='#e74c3c', label='Eliminated')
        ax.legend(handles=[survived_patch, eliminated_patch], loc='upper left')

        ax.set_title("Evolution Family Tree\nParent-Child Relationships and Survival",
                    fontsize=14, weight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'family_tree.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _create_performance_timeline(self):
        """Create timeline showing performance trends"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        generations = []
        best_accuracy = []
        avg_accuracy = []
        best_perplexity = []
        avg_perplexity = []

        for gen_data in self.history:
            generations.append(gen_data['generation'])
            best_accuracy.append(gen_data['best_accuracy'])
            avg_accuracy.append(gen_data['avg_accuracy'])
            best_perplexity.append(gen_data['best_perplexity'])

            # Calculate average perplexity
            perplexities = [v['eval_perplexity'] for v in gen_data['variants']]
            avg_perplexity.append(np.mean(perplexities))

        # Accuracy plot
        ax1.plot(generations, best_accuracy, 'o-', color='#2ecc71',
                linewidth=2, label='Best Accuracy', markersize=8)
        ax1.plot(generations, avg_accuracy, 's-', color='#3498db',
                linewidth=1.5, label='Average Accuracy', markersize=6, alpha=0.7)
        ax1.fill_between(generations, avg_accuracy, best_accuracy,
                         alpha=0.2, color='#2ecc71')

        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.set_title('Accuracy Evolution Over Generations', fontsize=12, weight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        # Perplexity plot
        ax2.plot(generations, best_perplexity, 'o-', color='#e74c3c',
                linewidth=2, label='Best Perplexity', markersize=8)
        ax2.plot(generations, avg_perplexity, 's-', color='#f39c12',
                linewidth=1.5, label='Average Perplexity', markersize=6, alpha=0.7)
        ax2.fill_between(generations, best_perplexity, avg_perplexity,
                         alpha=0.2, color='#e74c3c')

        ax2.set_xlabel('Generation', fontsize=11)
        ax2.set_ylabel('Perplexity', fontsize=11)
        ax2.set_title('Perplexity Evolution Over Generations', fontsize=12, weight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'performance_timeline.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _create_hyperparameter_heatmap(self):
        """Create heatmap showing hyperparameter effectiveness"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Prepare data for heatmaps
        ranks = sorted(set(v['rank'] for v in self.variants.values()))
        lrs = sorted(set(v['learning_rate'] for v in self.variants.values()))
        dropouts = sorted(set(v['dropout'] for v in self.variants.values()))

        # Rank vs LR heatmap
        rank_lr_matrix = np.zeros((len(ranks), len(lrs)))
        rank_lr_counts = np.zeros((len(ranks), len(lrs)))

        for variant in self.variants.values():
            r_idx = ranks.index(variant['rank'])
            lr_idx = lrs.index(variant['learning_rate'])
            rank_lr_matrix[r_idx, lr_idx] += variant['accuracy']
            rank_lr_counts[r_idx, lr_idx] += 1

        # Average where we have data
        mask = rank_lr_counts > 0
        rank_lr_matrix[mask] /= rank_lr_counts[mask]
        rank_lr_matrix[~mask] = np.nan

        sns.heatmap(rank_lr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=[f'{lr:.0e}' for lr in lrs],
                   yticklabels=ranks, ax=axes[0], cbar_kws={'label': 'Accuracy'})
        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Rank')
        axes[0].set_title('Accuracy by Rank vs Learning Rate')

        # Rank vs Dropout heatmap
        rank_dropout_matrix = np.zeros((len(ranks), len(dropouts)))
        rank_dropout_counts = np.zeros((len(ranks), len(dropouts)))

        for variant in self.variants.values():
            r_idx = ranks.index(variant['rank'])
            d_idx = dropouts.index(variant['dropout'])
            rank_dropout_matrix[r_idx, d_idx] += variant['accuracy']
            rank_dropout_counts[r_idx, d_idx] += 1

        mask = rank_dropout_counts > 0
        rank_dropout_matrix[mask] /= rank_dropout_counts[mask]
        rank_dropout_matrix[~mask] = np.nan

        sns.heatmap(rank_dropout_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=dropouts, yticklabels=ranks, ax=axes[1],
                   cbar_kws={'label': 'Accuracy'})
        axes[1].set_xlabel('Dropout')
        axes[1].set_ylabel('Rank')
        axes[1].set_title('Accuracy by Rank vs Dropout')

        # LR vs Dropout heatmap
        lr_dropout_matrix = np.zeros((len(lrs), len(dropouts)))
        lr_dropout_counts = np.zeros((len(lrs), len(dropouts)))

        for variant in self.variants.values():
            lr_idx = lrs.index(variant['learning_rate'])
            d_idx = dropouts.index(variant['dropout'])
            lr_dropout_matrix[lr_idx, d_idx] += variant['accuracy']
            lr_dropout_counts[lr_idx, d_idx] += 1

        mask = lr_dropout_counts > 0
        lr_dropout_matrix[mask] /= lr_dropout_counts[mask]
        lr_dropout_matrix[~mask] = np.nan

        sns.heatmap(lr_dropout_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=dropouts, yticklabels=[f'{lr:.0e}' for lr in lrs],
                   ax=axes[2], cbar_kws={'label': 'Accuracy'})
        axes[2].set_xlabel('Dropout')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Accuracy by Learning Rate vs Dropout')

        plt.suptitle('Hyperparameter Effectiveness Heatmaps', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'hyperparameter_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _create_survival_analysis(self):
        """Create survival analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Survival rate by hyperparameter
        hyperparams = ['rank', 'learning_rate', 'dropout']

        for i, param in enumerate(hyperparams):
            ax = axes[i // 2, i % 2]

            # Get unique values
            values = sorted(set(v[param] for v in self.variants.values()))
            survival_rates = []
            total_counts = []

            for val in values:
                variants_with_val = [v for v in self.variants.values()
                                    if v[param] == val]
                survived = sum(1 for v in variants_with_val if v['survived'])
                total = len(variants_with_val)
                survival_rates.append(survived / total if total > 0 else 0)
                total_counts.append(total)

            # Create bar chart
            bars = ax.bar(range(len(values)), survival_rates, color='#3498db', alpha=0.7)

            # Add count labels on bars
            for j, (bar, count) in enumerate(zip(bars, total_counts)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'n={count}', ha='center', va='bottom', fontsize=8)

            # Format x-axis labels
            if param == 'learning_rate':
                ax.set_xticklabels([f'{v:.0e}' for v in values], rotation=45)
            else:
                ax.set_xticklabels([str(v) for v in values])

            ax.set_xticks(range(len(values)))
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel('Survival Rate')
            ax.set_title(f'Survival Rate by {param.replace("_", " ").title()}')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')

        # Mutation effectiveness
        ax = axes[1, 1]

        mutation_types = list(self.mutation_success.keys())
        success_rates = []
        totals = []

        for mut_type in mutation_types:
            total = self.mutation_success[mut_type]['total']
            survived = self.mutation_success[mut_type]['survived']
            success_rates.append(survived / total if total > 0 else 0)
            totals.append(total)

        bars = ax.bar(range(len(mutation_types)), success_rates, color='#2ecc71', alpha=0.7)

        # Add count labels
        for j, (bar, total) in enumerate(zip(bars, totals)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={total}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(range(len(mutation_types)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in mutation_types])
        ax.set_ylabel('Survival Rate')
        ax.set_title('Mutation Survival Rate by Type')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Survival Analysis', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'survival_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _create_mutation_effectiveness(self):
        """Create detailed mutation effectiveness analysis"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Analyze mutation paths
        mutation_paths = []

        for variant_id, variant_data in self.variants.items():
            parent_id = variant_data.get('parent_id')
            if parent_id and parent_id in self.variants:
                parent = self.variants[parent_id]

                # Track what changed and performance delta
                mutations = []
                if variant_data['rank'] != parent['rank']:
                    mutations.append(f"Rank: {parent['rank']}→{variant_data['rank']}")
                if variant_data['learning_rate'] != parent['learning_rate']:
                    mutations.append(f"LR: {parent['learning_rate']:.0e}→{variant_data['learning_rate']:.0e}")
                if variant_data['dropout'] != parent['dropout']:
                    mutations.append(f"Dropout: {parent['dropout']}→{variant_data['dropout']}")

                if mutations:
                    accuracy_delta = variant_data['accuracy'] - parent['accuracy']
                    mutation_paths.append({
                        'generation': variant_data['generation'],
                        'mutations': ', '.join(mutations),
                        'accuracy_delta': accuracy_delta,
                        'survived': variant_data['survived']
                    })

        # Group by mutation type and calculate statistics
        mutation_stats = {}
        for path in mutation_paths:
            mut_key = path['mutations']
            if mut_key not in mutation_stats:
                mutation_stats[mut_key] = {
                    'deltas': [],
                    'survived_count': 0,
                    'total_count': 0
                }
            mutation_stats[mut_key]['deltas'].append(path['accuracy_delta'])
            mutation_stats[mut_key]['total_count'] += 1
            if path['survived']:
                mutation_stats[mut_key]['survived_count'] += 1

        # Sort by average improvement
        sorted_mutations = sorted(
            mutation_stats.items(),
            key=lambda x: np.mean(x[1]['deltas']),
            reverse=True
        )[:20]  # Top 20 mutation types

        # Create horizontal bar chart
        mutation_names = []
        avg_improvements = []
        survival_rates = []

        for mut_name, stats in sorted_mutations:
            # Shorten mutation names for display
            short_name = mut_name.replace('Rank: ', 'R')
            short_name = short_name.replace('LR: ', 'L')
            short_name = short_name.replace('Dropout: ', 'D')
            mutation_names.append(short_name)
            avg_improvements.append(np.mean(stats['deltas']))
            survival_rates.append(stats['survived_count'] / stats['total_count'])

        y_pos = np.arange(len(mutation_names))

        # Create bars with color based on survival rate
        colors = plt.cm.RdYlGn(np.array(survival_rates))
        bars = ax.barh(y_pos, avg_improvements, color=colors, alpha=0.8)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, avg_improvements)):
            width = bar.get_width()
            label_x = width + 0.001 if width >= 0 else width - 0.001
            ha = 'left' if width >= 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                   f'{val:.1%}', ha=ha, va='center', fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(mutation_names, fontsize=9)
        ax.set_xlabel('Average Accuracy Improvement', fontsize=11)
        ax.set_title('Mutation Effectiveness (Top 20 Mutation Types)\nColor indicates survival rate',
                    fontsize=12, weight='bold')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='x')

        # Add colorbar for survival rate
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Survival Rate', rotation=270, labelpad=20)

        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'mutation_effectiveness.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_report(self):
        """Generate comprehensive markdown report"""
        report_path = self.analysis_dir / 'evolution_analysis_report.md'

        with open(report_path, 'w') as f:
            f.write("# Evolution Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overview
            f.write("## Overview\n\n")
            f.write(f"- **Total Generations**: {len(self.history)}\n")
            f.write(f"- **Total Variants Created**: {len(self.variants)}\n")
            f.write(f"- **Total Survivors**: {sum(1 for v in self.variants.values() if v['survived'])}\n")

            # Best variant
            best_variant = max(self.variants.values(), key=lambda v: v['fitness'])
            f.write(f"\n### Best Variant Overall\n\n")
            f.write(f"- **Generation**: {best_variant['generation']}\n")
            f.write(f"- **Rank**: {best_variant['rank']}\n")
            f.write(f"- **Learning Rate**: {best_variant['learning_rate']:.0e}\n")
            f.write(f"- **Dropout**: {best_variant['dropout']}\n")
            f.write(f"- **Accuracy**: {best_variant['accuracy']:.2%}\n")
            f.write(f"- **Perplexity**: {best_variant['perplexity']:.2f}\n")
            f.write(f"- **Fitness Score**: {best_variant['fitness']:.4f}\n")

            # Performance progression
            f.write("\n## Performance Progression\n\n")
            f.write("| Generation | Best Accuracy | Avg Accuracy | Best Perplexity | Survivors |\n")
            f.write("|------------|---------------|--------------|-----------------|----------|\n")

            for gen_data in self.history:
                gen = gen_data['generation']
                survivors = self.survivors_by_gen.get(gen, [])
                survivor_str = ', '.join([s.split('_')[0] for s in survivors[:2]])
                f.write(f"| {gen:10d} | {gen_data['best_accuracy']:13.2%} | "
                       f"{gen_data['avg_accuracy']:12.2%} | {gen_data['best_perplexity']:15.2f} | "
                       f"{survivor_str:9s} |\n")

            # Mutation effectiveness
            f.write("\n## Mutation Effectiveness\n\n")
            f.write("| Mutation Type | Total | Survived | Success Rate |\n")
            f.write("|---------------|-------|----------|-------------|\n")

            for mut_type, stats in self.mutation_success.items():
                total = stats['total']
                survived = stats['survived']
                rate = survived / total if total > 0 else 0
                f.write(f"| {mut_type:13s} | {total:5d} | {survived:8d} | {rate:11.2%} |\n")

            # Hyperparameter insights
            f.write("\n## Hyperparameter Insights\n\n")

            # Best performing values
            for param in ['rank', 'learning_rate', 'dropout']:
                values = {}
                for v in self.variants.values():
                    val = v[param]
                    if val not in values:
                        values[val] = []
                    values[val].append(v['accuracy'])

                best_val = max(values.keys(), key=lambda x: np.mean(values[x]))
                avg_acc = np.mean(values[best_val])

                if param == 'learning_rate':
                    f.write(f"- **Best {param.replace('_', ' ').title()}**: {best_val:.0e} "
                           f"(avg accuracy: {avg_acc:.2%})\n")
                else:
                    f.write(f"- **Best {param.replace('_', ' ').title()}**: {best_val} "
                           f"(avg accuracy: {avg_acc:.2%})\n")

            # Lineage insights
            f.write("\n## Lineage Analysis\n\n")

            # Find longest successful lineage
            lineages = {}
            for variant_id, parent_id in self.lineage.items():
                if parent_id not in lineages:
                    lineages[parent_id] = []
                lineages[parent_id].append(variant_id)

            # Count successful generations
            successful_chains = []
            for start_id in self.variants.keys():
                if start_id not in self.lineage.values():  # Is a root
                    chain = [start_id]
                    current = start_id
                    while current in lineages:
                        children = lineages[current]
                        # Find best surviving child
                        surviving_children = [c for c in children
                                            if c in self.variants and
                                            self.variants[c]['survived']]
                        if surviving_children:
                            best_child = max(surviving_children,
                                          key=lambda c: self.variants[c]['fitness'])
                            chain.append(best_child)
                            current = best_child
                        else:
                            break
                    if len(chain) > 1:
                        successful_chains.append(chain)

            if successful_chains:
                longest_chain = max(successful_chains, key=len)
                f.write(f"- **Longest Successful Lineage**: {len(longest_chain)} generations\n")
                f.write(f"- **Starting Variant**: {longest_chain[0]}\n")
                f.write(f"- **Final Variant**: {longest_chain[-1]}\n")

            f.write("\n## Visualization Files\n\n")
            f.write("The following visualizations have been generated:\n\n")
            f.write("1. `family_tree.png` - Evolution family tree showing parent-child relationships\n")
            f.write("2. `performance_timeline.png` - Performance trends over generations\n")
            f.write("3. `hyperparameter_heatmap.png` - Hyperparameter combination effectiveness\n")
            f.write("4. `survival_analysis.png` - Survival rates by hyperparameter\n")
            f.write("5. `mutation_effectiveness.png` - Mutation type effectiveness analysis\n")

        logger.info(f"Report generated: {report_path}")


def main():
    """CLI entry point for analysis"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze LoRA evolution history")
    parser.add_argument(
        '--dir',
        default='evolved_adapters',
        help='Path to evolution output directory'
    )

    args = parser.parse_args()

    analyzer = EvolutionAnalyzer(args.dir)
    analyzer.analyze()


if __name__ == '__main__':
    main()