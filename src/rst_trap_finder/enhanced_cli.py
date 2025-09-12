"""Enhanced CLI with rich formatting and interactive features."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import click
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track, Progress
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    click = None

from . import TRAP_LETTERS
from .io import load_csv
from .data_processing import DataProcessor
from .scores import biased_pagerank, composite, one_step_rst_prob, escape_hardness, k_step_rst_prob, minimax_topm
from .strategy import recommend_next

if RICH_AVAILABLE:
    console = Console()
else:
    console = None


class Config:
    """Configuration management for CLI."""
    
    def __init__(self):
        self.config_path = Path.home() / '.rst_trap_finder' / 'config.json'
        self.config_path.parent.mkdir(exist_ok=True)
        self._config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        
        # Default configuration
        return {
            'default_lambdas': [0.35, 0.2, 0.25, 0.1, 0.1],
            'default_alpha': 1.5,
            'default_top': 15,
            'output_format': 'table',
            'show_progress': True,
            'auto_save_results': False,
            'results_dir': str(Path.home() / 'rst_analysis_results'),
        }
    
    def save_config(self):
        """Save configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value."""
        self._config[key] = value
        self.save_config()


config = Config()


def print_rich_table(data: List[Dict], headers: List[str], title: str = "Results"):
    """Print a rich formatted table."""
    if not RICH_AVAILABLE:
        # Fallback to basic formatting
        print(f"\n{title}")
        print("-" * len(title))
        
        # Simple table formatting
        col_widths = {h: len(h) for h in headers}
        for row in data:
            for h in headers:
                col_widths[h] = max(col_widths[h], len(str(row.get(h, ''))))
        
        # Header
        header_row = " | ".join(h.ljust(col_widths[h]) for h in headers)
        print(header_row)
        print("-" * len(header_row))
        
        # Data rows
        for row in data:
            data_row = " | ".join(str(row.get(h, '')).ljust(col_widths[h]) for h in headers)
            print(data_row)
        
        return
    
    table = Table(title=title, show_header=True, header_style="bold magenta")
    
    for header in headers:
        table.add_column(header)
    
    for row in data:
        row_data = [str(row.get(h, '')) for h in headers]
        table.add_row(*row_data)
    
    console.print(table)


def print_info_panel(title: str, content: str):
    """Print an information panel."""
    if not RICH_AVAILABLE:
        print(f"\n{title}")
        print("=" * len(title))
        print(content)
        return
    
    console.print(Panel(content, title=title, expand=False))


if click is None:
    # Fallback when click is not available
    def main(argv: List[str] | None = None):
        print("Error: This CLI requires 'click' and 'rich' packages to be installed.")
        print("Install with: pip install click rich")
        sys.exit(1)
else:
    @click.group()
    @click.option('--config-file', type=click.Path(), help='Path to configuration file')
    @click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
    @click.pass_context
    def cli(ctx, config_file, verbose):
        """RST Trap Finder - Comprehensive word graph analysis toolkit."""
        ctx.ensure_object(dict)
        ctx.obj['verbose'] = verbose
        
        if config_file:
            # Load custom config file
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                config._config.update(custom_config)
            except Exception as e:
                if RICH_AVAILABLE:
                    console.print(f"[red]Error loading config file: {e}[/red]")
                else:
                    print(f"Error loading config file: {e}")
    
    
    @cli.command()
    @click.option('--csv', required=True, type=click.Path(exists=True), help='Input CSV file')
    @click.option('--format', 'input_format', default='csv', 
                  type=click.Choice(['csv', 'json', 'pickle']), help='Input format')
    @click.option('--top', default=None, type=int, help='Number of top results to show')
    @click.option('--lambdas', default=None, help='Comma-separated lambda values for composite scoring')
    @click.option('--alpha', default=None, type=float, help='PageRank bias parameter')
    @click.option('--output', '-o', type=click.Path(), help='Save results to file')
    @click.option('--output-format', type=click.Choice(['table', 'csv', 'json']), 
                  default=None, help='Output format')
    @click.option('--include-advanced', is_flag=True, help='Include advanced scoring metrics')
    @click.pass_context
    def rank(ctx, csv, input_format, top, lambdas, alpha, output, output_format, include_advanced):
        """Rank words by trap effectiveness."""
        verbose = ctx.obj.get('verbose', False)
        
        # Load configuration
        top = top or config.get('default_top', 15)
        alpha = alpha or config.get('default_alpha', 1.5)
        output_format = output_format or config.get('output_format', 'table')
        
        if lambdas:
            lamb = tuple(float(x) for x in lambdas.split(','))
            if len(lamb) != 5:
                raise click.BadParameter('Lambdas must be 5 comma-separated values')
        else:
            lamb = tuple(config.get('default_lambdas', [0.35, 0.2, 0.25, 0.1, 0.1]))
        
        if verbose:
            print_info_panel("Configuration", 
                           f"Input: {csv}\nFormat: {input_format}\nTop: {top}\n" +
                           f"Alpha: {alpha}\nLambdas: {lamb}")
        
        # Load graph
        if RICH_AVAILABLE and config.get('show_progress', True):
            with Progress() as progress:
                task = progress.add_task("Loading graph...", total=100)
                if input_format == 'csv':
                    graph = DataProcessor.load_csv(csv)
                elif input_format == 'json':
                    graph = DataProcessor.load_json(csv)
                elif input_format == 'pickle':
                    graph = DataProcessor.load_pickle(csv)
                progress.update(task, completed=100)
        else:
            if input_format == 'csv':
                graph = DataProcessor.load_csv(csv)
            elif input_format == 'json':
                graph = DataProcessor.load_json(csv)
            elif input_format == 'pickle':
                graph = DataProcessor.load_pickle(csv)
        
        # Compute scores
        if RICH_AVAILABLE and config.get('show_progress', True):
            console.print("Computing PageRank...")
        
        pr = biased_pagerank(graph, TRAP_LETTERS, alpha)
        
        # Prepare results
        results = []
        nodes_to_process = list(graph.keys())
        
        if RICH_AVAILABLE and config.get('show_progress', True):
            nodes_to_process = track(nodes_to_process, description="Computing scores...")
        
        for node in nodes_to_process:
            row = {
                'word': node,
                'composite': f"{composite(node, graph, TRAP_LETTERS, pr, lamb):.3f}",
                'one_step': f"{one_step_rst_prob(node, graph, TRAP_LETTERS):.3f}",
                'escape_hardness': f"{escape_hardness(node, graph, TRAP_LETTERS):.3f}",
                'pagerank': f"{pr.get(node, 0.0):.3e}",
                'k2_step': f"{k_step_rst_prob(node, graph, TRAP_LETTERS, k=2):.3f}",
                'minimax': f"{minimax_topm(node, graph, TRAP_LETTERS):.3f}",
                'out_degree': len(graph.get(node, {})),
            }
            
            if include_advanced:
                try:
                    from .advanced_scores import comprehensive_score
                    row['advanced_score'] = f"{comprehensive_score(node, graph, TRAP_LETTERS):.3f}"
                except ImportError:
                    if verbose:
                        print_info_panel("Warning", "Advanced scoring not available (missing dependencies)")
            
            results.append(row)
        
        # Sort by composite score
        results.sort(key=lambda r: float(r['composite']), reverse=True)
        results = results[:top]
        
        # Output results
        headers = ['word', 'composite', 'one_step', 'escape_hardness', 'pagerank', 'k2_step', 'minimax', 'out_degree']
        if include_advanced and 'advanced_score' in results[0]:
            headers.append('advanced_score')
        
        if output_format == 'table':
            print_rich_table(results, headers, f"Top {top} Trap Words")
        elif output_format == 'csv':
            import pandas as pd
            df = pd.DataFrame(results)
            if output:
                df.to_csv(output, index=False)
                print_info_panel("Success", f"Results saved to {output}")
            else:
                print(df.to_csv(index=False))
        elif output_format == 'json':
            if output:
                with open(output, 'w') as f:
                    json.dump(results, f, indent=2)
                print_info_panel("Success", f"Results saved to {output}")
            else:
                print(json.dumps(results, indent=2))
    
    
    @cli.command()
    @click.option('--word', required=True, help='Current word')
    @click.option('--csv', required=True, type=click.Path(exists=True), help='Input CSV file')
    @click.option('--lambdas', default=None, help='Comma-separated lambda values')
    @click.option('--show-paths', is_flag=True, help='Show paths to trap words')
    @click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
    @click.pass_context
    def next(ctx, word, csv, lambdas, show_paths, interactive):
        """Recommend next word from current position."""
        verbose = ctx.obj.get('verbose', False)
        
        if lambdas:
            lamb = tuple(float(x) for x in lambdas.split(','))
        else:
            lamb = tuple(config.get('default_lambdas'))
        
        graph = load_csv(csv)
        pr = biased_pagerank(graph)
        
        try:
            result = recommend_next(word, graph, TRAP_LETTERS, pr, lamb)
        except ValueError as e:
            if RICH_AVAILABLE:
                console.print(f"[red]Error: {e}[/red]")
            else:
                print(f"Error: {e}")
            return
        
        best = result['best']
        print_info_panel("Best Recommendation", f"Word: {best['word']}")
        
        # Show candidates table
        candidates_data = []
        for c in result['candidates'][:10]:  # Top 10
            candidates_data.append({
                'word': c['word'],
                'composite': f"{c['composite']:.3f}",
                'basin': f"{c['basin']:.3f}",
                'exits': c['non_rst_strong_exits'],
                'expected': f"{c['expected']:.3f}",
            })
        
        headers = ['word', 'composite', 'basin', 'exits', 'expected']
        print_rich_table(candidates_data, headers, "Candidate Words")
        
        if show_paths:
            # Show paths to trap words
            try:
                from .analysis import GraphAnalyzer
                analyzer = GraphAnalyzer(graph, TRAP_LETTERS)
                paths = analyzer.path_analysis(word)
                
                if RICH_AVAILABLE:
                    tree = Tree("Paths to Trap Words")
                    for target, target_paths in paths.items():
                        if target_paths:
                            target_node = tree.add(f"[red]{target}[/red]")
                            for i, path in enumerate(target_paths[:3]):  # Show top 3 paths
                                path_str = " → ".join(path)
                                target_node.add(f"Path {i+1}: {path_str}")
                    console.print(tree)
                else:
                    print("\nPaths to Trap Words:")
                    for target, target_paths in paths.items():
                        if target_paths:
                            print(f"\n{target}:")
                            for i, path in enumerate(target_paths[:3]):
                                path_str = " → ".join(path)
                                print(f"  Path {i+1}: {path_str}")
            except ImportError:
                if verbose:
                    print_info_panel("Warning", "Path analysis not available (missing dependencies)")
        
        if interactive:
            while True:
                if RICH_AVAILABLE:
                    next_word = Prompt.ask("Enter next word (or 'quit' to exit)")
                else:
                    next_word = input("Enter next word (or 'quit' to exit): ")
                
                if next_word.lower() in ['quit', 'exit', 'q']:
                    break
                
                if next_word in graph:
                    # Recursive call with new word
                    ctx.invoke(next, word=next_word, csv=csv, lambdas=lambdas, 
                             show_paths=show_paths, interactive=False)
                else:
                    if RICH_AVAILABLE:
                        console.print(f"[yellow]Warning: '{next_word}' not found in graph[/yellow]")
                    else:
                        print(f"Warning: '{next_word}' not found in graph")
    
    
    @cli.command()
    @click.option('--csv', required=True, type=click.Path(exists=True), help='Input CSV file')
    @click.option('--output-dir', type=click.Path(), help='Output directory for analysis files')
    @click.option('--include-viz', is_flag=True, help='Include visualizations')
    @click.option('--include-ml', is_flag=True, help='Include ML analysis')
    @click.pass_context
    def analyze(ctx, csv, output_dir, include_viz, include_ml):
        """Comprehensive graph analysis."""
        verbose = ctx.obj.get('verbose', False)
        
        if not output_dir:
            output_dir = Path(config.get('results_dir', '.')) / 'analysis'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        graph = load_csv(csv)
        
        if verbose:
            print_info_panel("Analysis Configuration", 
                           f"Input: {csv}\nOutput: {output_dir}\n" +
                           f"Visualizations: {include_viz}\nML Analysis: {include_ml}")
        
        # Basic analysis
        try:
            from .analysis import GraphAnalyzer
            analyzer = GraphAnalyzer(graph, TRAP_LETTERS)
            
            if RICH_AVAILABLE:
                console.print("Performing basic graph analysis...")
            
            # Export basic analysis
            analyzer.export_analysis(output_dir, prefix="basic")
            
            # Print summary
            stats = analyzer.basic_stats()
            stats_text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
            print_info_panel("Graph Statistics", stats_text)
            
        except ImportError:
            if RICH_AVAILABLE:
                console.print("[yellow]Skipping graph analysis (missing dependencies)[/yellow]")
            else:
                print("Skipping graph analysis (missing dependencies)")
        
        # ML Analysis
        if include_ml:
            try:
                from .ml_models import TrapPredictor, FeatureExtractor
                
                if RICH_AVAILABLE:
                    console.print("Training ML models...")
                
                # Train predictor
                predictor = TrapPredictor()
                metrics = predictor.train(graph)
                
                # Save model
                model_path = output_dir / "trap_predictor.pkl"
                predictor.save_model(model_path)
                
                # Save metrics
                with open(output_dir / "ml_metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                print_info_panel("ML Training Results", 
                               f"Train Score: {metrics['train_score']:.3f}\n" +
                               f"Test Score: {metrics['test_score']:.3f}")
                
            except ImportError:
                if RICH_AVAILABLE:
                    console.print("[yellow]Skipping ML analysis (missing dependencies)[/yellow]")
                else:
                    print("Skipping ML analysis (missing dependencies)")
        
        # Visualizations
        if include_viz:
            try:
                from .analysis import GraphAnalyzer
                analyzer = GraphAnalyzer(graph, TRAP_LETTERS)
                
                if RICH_AVAILABLE:
                    console.print("Creating visualizations...")
                
                # Create network plot
                fig = analyzer.visualize_network(save_path=output_dir / "network.png")
                
                # Create interactive plot
                interactive_fig = analyzer.interactive_network()
                interactive_fig.write_html(output_dir / "interactive_network.html")
                
                print_info_panel("Visualizations", 
                               f"Network plot: {output_dir}/network.png\n" +
                               f"Interactive: {output_dir}/interactive_network.html")
                
            except ImportError:
                if RICH_AVAILABLE:
                    console.print("[yellow]Skipping visualizations (missing dependencies)[/yellow]")
                else:
                    print("Skipping visualizations (missing dependencies)")
        
        print_info_panel("Analysis Complete", f"Results saved to: {output_dir}")
    
    
    @cli.command()
    @click.option('--key', help='Configuration key to set/get')
    @click.option('--value', help='Configuration value to set')
    @click.option('--list-all', is_flag=True, help='List all configuration')
    @click.option('--reset', is_flag=True, help='Reset to default configuration')
    def config_cmd(key, value, list_all, reset):
        """Manage configuration settings."""
        if reset:
            if RICH_AVAILABLE:
                if Confirm.ask("Reset all configuration to defaults?"):
                    config._config = config.load_config()
                    config.save_config()
                    console.print("[green]Configuration reset to defaults[/green]")
            else:
                response = input("Reset all configuration to defaults? (y/N): ")
                if response.lower().startswith('y'):
                    config._config = config.load_config()
                    config.save_config()
                    print("Configuration reset to defaults")
            return
        
        if list_all:
            config_text = json.dumps(config._config, indent=2)
            if RICH_AVAILABLE:
                syntax = Syntax(config_text, "json", theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title="Current Configuration"))
            else:
                print("Current Configuration:")
                print(config_text)
            return
        
        if key and value:
            # Set configuration
            try:
                # Try to parse as JSON for complex values
                parsed_value = json.loads(value)
            except json.JSONDecodeError:
                # Use as string
                parsed_value = value
            
            config.set(key, parsed_value)
            if RICH_AVAILABLE:
                console.print(f"[green]Set {key} = {parsed_value}[/green]")
            else:
                print(f"Set {key} = {parsed_value}")
        
        elif key:
            # Get configuration
            value = config.get(key)
            if RICH_AVAILABLE:
                console.print(f"{key}: [cyan]{value}[/cyan]")
            else:
                print(f"{key}: {value}")
        
        else:
            if RICH_AVAILABLE:
                console.print("[red]Please specify --key and --value to set, or --key to get[/red]")
            else:
                print("Please specify --key and --value to set, or --key to get")
    
    
    @cli.command()
    @click.option('--input-file', required=True, type=click.Path(exists=True), help='Input file')
    @click.option('--input-format', type=click.Choice(['csv', 'json', 'pickle']), 
                  default='csv', help='Input format')
    @click.option('--output-file', required=True, type=click.Path(), help='Output file')
    @click.option('--output-format', type=click.Choice(['csv', 'json', 'pickle', 'graphml']),
                  required=True, help='Output format')
    def convert(input_file, input_format, output_file, output_format):
        """Convert between different graph formats."""
        # Load graph
        if input_format == 'csv':
            graph = DataProcessor.load_csv(input_file)
        elif input_format == 'json':
            graph = DataProcessor.load_json(input_file)
        elif input_format == 'pickle':
            graph = DataProcessor.load_pickle(input_file)
        
        # Save graph
        try:
            if output_format == 'csv':
                DataProcessor.save_csv(graph, output_file)
            elif output_format == 'json':
                DataProcessor.save_json(graph, output_file)
            elif output_format == 'pickle':
                DataProcessor.save_pickle(graph, output_file)
            elif output_format == 'graphml':
                DataProcessor.save_graphml(graph, output_file)
            
            if RICH_AVAILABLE:
                console.print(f"[green]Converted {input_file} to {output_file}[/green]")
            else:
                print(f"Converted {input_file} to {output_file}")
        
        except ImportError as e:
            if RICH_AVAILABLE:
                console.print(f"[red]Error: {e}[/red]")
            else:
                print(f"Error: {e}")
    
    
    def main(argv: List[str] | None = None):
        """Main entry point."""
        if not RICH_AVAILABLE:
            print("Warning: Rich formatting not available. Install with: pip install rich")
        
        cli(argv, standalone_mode=False)