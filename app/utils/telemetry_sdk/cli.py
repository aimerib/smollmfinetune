"""
Telemetry SDK CLI Reporter

Command-line interface for querying and displaying experiment run information.

Usage:
    python -m telemetry_sdk.report run_id
"""

import argparse
import sqlite3
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def format_run_report(run_id: str, db_path: str = 'telemetry.db') -> str:
    """
    Format a human-readable report for a specific run.
    
    Args:
        run_id: Run identifier to report on
        db_path: Path to SQLite database
        
    Returns:
        Formatted report string
    """
    db_file = Path(db_path)
    if not db_file.exists():
        return f"Database not found: {db_path}"
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get run information
            cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
            run_row = cursor.fetchone()
            
            if not run_row:
                return f"Run not found: {run_id}"
            
            # Parse run data
            run_data = {
                'id': run_row[0],
                'started_at': run_row[1],
                'git_sha': run_row[2],
                'config_json': run_row[3],
                'notes': run_row[4],
                'pip_freeze': run_row[5],
                'run_env_vars': run_row[6]
            }
            
            # Get metrics summary
            cursor.execute("""
                SELECT key, COUNT(*) as count, MIN(value) as min_val, 
                       MAX(value) as max_val, AVG(value) as avg_val
                FROM metrics 
                WHERE run_id = ? 
                GROUP BY key
                ORDER BY key
            """, (run_id,))
            
            metrics_rows = cursor.fetchall()
            
            # Format report
            report_lines = [
                "=" * 60,
                f"EXPERIMENT RUN REPORT: {run_id}",
                "=" * 60,
                "",
                f"Started: {run_data['started_at']}",
                f"Git SHA: {run_data['git_sha'][:12]}..." if run_data['git_sha'] != 'unknown' else "Git SHA: unknown",
                f"Notes: {run_data['notes']}",
                "",
                "CONFIGURATION:",
                "-" * 20
            ]
            
            # Format configuration
            try:
                config = json.loads(run_data['config_json'])
                for key, value in config.items():
                    report_lines.append(f"  {key}: {value}")
            except json.JSONDecodeError:
                report_lines.append("  (Configuration parsing error)")
            
            report_lines.extend(["", "ENVIRONMENT VARIABLES:", "-" * 25])
            
            # Format environment variables
            try:
                env_vars = json.loads(run_data['run_env_vars'] or '{}')
                if env_vars:
                    for key, value in env_vars.items():
                        report_lines.append(f"  {key}: {value}")
                else:
                    report_lines.append("  (No RUN_ environment variables)")
            except json.JSONDecodeError:
                report_lines.append("  (Environment variables parsing error)")
            
            report_lines.extend(["", "METRICS SUMMARY:", "-" * 20])
            
            if metrics_rows:
                # Header
                report_lines.append(f"{'Metric':<20} {'Count':<8} {'Min':<12} {'Max':<12} {'Avg':<12}")
                report_lines.append("-" * 68)
                
                # Metrics data
                for row in metrics_rows:
                    key, count, min_val, max_val, avg_val = row
                    report_lines.append(
                        f"{key:<20} {count:<8} {min_val:<12.4f} {max_val:<12.4f} {avg_val:<12.4f}"
                    )
            else:
                report_lines.append("  (No metrics logged)")
            
            # Add dependency information if available
            if run_data['pip_freeze'] and run_data['pip_freeze'] != 'unknown':
                report_lines.extend(["", "KEY DEPENDENCIES:", "-" * 20])
                pip_lines = run_data['pip_freeze'].split('\n')
                key_packages = ['torch', 'transformers', 'numpy', 'pandas', 'trl', 'peft']
                
                for line in pip_lines:
                    if any(pkg in line.lower() for pkg in key_packages):
                        report_lines.append(f"  {line.strip()}")
            
            report_lines.append("=" * 60)
            
            return '\n'.join(report_lines)
            
    except sqlite3.Error as e:
        return f"Database error: {e}"
    except Exception as e:
        return f"Error generating report: {e}"


def list_runs(db_path: str = 'telemetry.db', limit: int = 10) -> str:
    """
    List recent experiment runs.
    
    Args:
        db_path: Path to SQLite database
        limit: Maximum number of runs to show
        
    Returns:
        Formatted list of runs
    """
    db_file = Path(db_path)
    if not db_file.exists():
        return f"Database not found: {db_path}"
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, started_at, notes, git_sha
                FROM runs 
                ORDER BY started_at DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            
            if not rows:
                return "No runs found in database."
            
            lines = [
                "RECENT EXPERIMENT RUNS:",
                "=" * 50,
                f"{'Run ID':<30} {'Started':<20} {'Git SHA':<12} {'Notes':<20}",
                "-" * 85
            ]
            
            for row in rows:
                run_id, started_at, notes, git_sha = row
                
                # Parse timestamp for display
                try:
                    dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    started_display = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    started_display = started_at[:16]
                
                git_display = git_sha[:8] if git_sha != 'unknown' else 'unknown'
                notes_display = (notes or '')[:18]
                
                lines.append(f"{run_id:<30} {started_display:<20} {git_display:<12} {notes_display:<20}")
            
            return '\n'.join(lines)
            
    except sqlite3.Error as e:
        return f"Database error: {e}"
    except Exception as e:
        return f"Error listing runs: {e}"


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Telemetry SDK - Experiment Run Reporter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m telemetry_sdk.report my_run_20241227_143022
  python -m telemetry_sdk.report --list
  python -m telemetry_sdk.report --list --db-path /path/to/custom.db
        """
    )
    
    parser.add_argument('run_id', nargs='?', help='Run ID to generate report for')
    parser.add_argument('--list', action='store_true', help='List recent runs')
    parser.add_argument('--db-path', default='telemetry.db', help='Path to SQLite database')
    parser.add_argument('--limit', type=int, default=10, help='Limit for --list option')
    
    args = parser.parse_args()
    
    if args.list:
        output = list_runs(args.db_path, args.limit)
    elif args.run_id:
        output = format_run_report(args.run_id, args.db_path)
    else:
        parser.print_help()
        return
    
    print(output)


if __name__ == '__main__':
    main() 