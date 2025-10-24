#!/usr/bin/env python3
"""
PII Detection Control Center - Comprehensive UI for PII Analysis System
Features:
- Run PII detection analysis with configurable settings
- Edit YAML configuration files
- View and analyze JSON results
- Real-time monitoring of analysis progress
- Export and download results
"""

import streamlit as st
import yaml
import json
import pandas as pd
from datetime import datetime
import subprocess
import os
import sys
from pathlib import Path
import time
import logging
from typing import Dict, List, Any, Optional, Set
import re
import difflib
import shutil
import ast
import importlib
from textwrap import dedent


# Import PII Exposure Analysis
from analyze_pii_exposure import PIIExposureAnalyzer

# Import the PII Exposure Report UI
try:
    from report_generator import display_exposure_report
    REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    REPORT_GENERATOR_AVAILABLE = False
    display_exposure_report = None

# Import Simple Presidio Anonymizer
try:
    from simple_presidio_anonymizer import process_csv
    SIMPLE_PRESIDIO_AVAILABLE = True
except ImportError:
    SIMPLE_PRESIDIO_AVAILABLE = False
    process_csv = None

# Try to import optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PII Detection Control Center",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit rerun compatibility helper (supports newer st.rerun and older st.experimental_rerun)
def _st_rerun():
    if hasattr(st, "rerun"):
        return st.rerun()
    if hasattr(st, "experimental_rerun"):
        return st.experimental_rerun()
    # As a last resort, do nothing; user can manually refresh
    return None

class PIIControlCenter:
    """Main control center for PII detection system"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.config_path = self.base_path / "config.yaml"
        self.analyzer_config_path = self.base_path / "analyzers_config.yaml"
        self.recognizers_config_path = self.base_path / "presidio_recognizers_config.yaml"
        # Backup directory for YAML versions
        self.backups_path = self.base_path / "_yaml_backups"
        self.backups_path.mkdir(exist_ok=True)

        # Create datasets and results folders if they don't exist
        self.datasets_path = self.base_path / "test_datasets"
        self.results_path = self.base_path / "results"
        self.datasets_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)
        
        # Initialize session state
        if 'analysis_running' not in st.session_state:
            st.session_state.analysis_running = False
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'config_modified' not in st.session_state:
            st.session_state.config_modified = False
        if 'analysis_start_time' not in st.session_state:
            st.session_state.analysis_start_time = None
        if 'analysis_params' not in st.session_state:
            st.session_state.analysis_params = None
    
    def _ensure_psutil(self) -> bool:
        """Attempt to import psutil at runtime."""
        global PSUTIL_AVAILABLE, psutil
        if PSUTIL_AVAILABLE and psutil is not None:
            return True
        try:
            psutil = importlib.import_module("psutil")  # type: ignore
            PSUTIL_AVAILABLE = True
            return True
        except Exception:
            PSUTIL_AVAILABLE = False
            psutil = None
            return False

    def _install_psutil(self) -> bool:
        """Install psutil via pip and refresh availability."""
        global PSUTIL_AVAILABLE, psutil
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "psutil"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                st.success("✅ psutil installed successfully. Reloading metrics…")
                return self._ensure_psutil()

            st.error("❌ psutil installation failed.")
            if result.stderr:
                st.code(result.stderr.strip(), language="bash")
            elif result.stdout:
                st.code(result.stdout.strip(), language="bash")
        except Exception as e:
            st.error(f"Failed to install psutil: {e}")
            logger.error("psutil installation error", exc_info=True)

        PSUTIL_AVAILABLE = False
        psutil = None
        return False

    def load_yaml_file(self, file_path: Path) -> Dict:
        """Load YAML file with error handling"""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            st.error(f"Error loading {file_path.name}: {e}")
            return {}

    def _list_available_yaml_recognizers(self) -> List[Dict[str, Any]]:
        """Parse YAML no-code recognizers and return a summary list.

        Uses the default `presidio_recognizers_config.yaml`. For ad-hoc files,
        prefer calling `_list_yaml_recognizers_from(path)`.
        """
        items: List[Dict[str, Any]] = []
        try:
            if self.recognizers_config_path.exists():
                cfg = self.load_yaml_file(self.recognizers_config_path)
                recs = cfg.get('recognizers', []) if isinstance(cfg, dict) else []
                for r in recs:
                    if not isinstance(r, dict):
                        continue
                    ents = r.get('supported_entity') or r.get('supported_entities') or []
                    if isinstance(ents, str):
                        ents_list = [ents]
                    elif isinstance(ents, list):
                        ents_list = [str(e) for e in ents]
                    else:
                        ents_list = []
                    items.append({
                        'name': r.get('name', 'Unnamed'),
                        'entities': ents_list,
                        'language': r.get('supported_language', 'en'),
                        'type': r.get('type', 'pattern'),
                    })
        except Exception:
            pass
        return items

    def _list_yaml_recognizers_from(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse YAML no-code recognizers from an explicit path and return a summary list."""
        items: List[Dict[str, Any]] = []
        try:
            if file_path.exists():
                cfg = self.load_yaml_file(file_path)
                recs = cfg.get('recognizers', []) if isinstance(cfg, dict) else []
                for r in recs:
                    if not isinstance(r, dict):
                        continue
                    ents = r.get('supported_entity') or r.get('supported_entities') or []
                    if isinstance(ents, str):
                        ents_list = [ents]
                    elif isinstance(ents, list):
                        ents_list = [str(e) for e in ents]
                    else:
                        ents_list = []
                    items.append({
                        'name': r.get('name', 'Unnamed'),
                        'entities': ents_list,
                        'language': r.get('supported_language', 'en'),
                        'type': r.get('type', 'pattern'),
                    })
        except Exception:
            pass
        return items

    def _find_yaml_candidates(self) -> List[Path]:
        """Find candidate YAML files in the workspace to use for recognizers."""
        roots = [self.base_path]
        # Common subfolders to look into
        for sub in ("config", "configs", "recognizers", "yaml", "presidio"):
            p = self.base_path / sub
            if p.exists() and p.is_dir():
                roots.append(p)
        seen: set = set()
        out: List[Path] = []
        for root in roots:
            for ext in ("*.yaml", "*.yml"):
                for fp in root.glob(ext):
                    key = str(fp.resolve())
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(fp)
        # Prefer files that look like recognizer configs first
        out.sort(key=lambda p: ("recognizer" not in p.name.lower() and "presidio" not in p.name.lower(), p.name.lower()))
        return out

    def _render_yaml_entries_preview(self, file_path: Path, selected_entities: Optional[List[str]] = None) -> None:
        """Render a detailed preview of individual YAML recognizers from a file.

        Shows name, entity, score, patterns (name + regex), and optional context keywords.
        Filters by selected_entities if provided and non-empty.
        """
        try:
            if not file_path or not file_path.exists():
                st.info("No YAML recognizers file found to preview.")
                return
            cfg = self.load_yaml_file(file_path)
            if not isinstance(cfg, dict):
                st.warning("YAML file did not parse to a recognizers config dict.")
                return
            recs = cfg.get('recognizers', []) or []
            if not isinstance(recs, list) or not recs:
                st.info("This YAML doesn't define any recognizers.")
                return
            # Filter by selected entities when provided
            filt: List[Dict[str, Any]] = []
            sel = set([e for e in (selected_entities or []) if e])
            for r in recs:
                if not isinstance(r, dict):
                    continue
                ents = r.get('supported_entity') or r.get('supported_entities')
                ents_list: List[str] = []
                if isinstance(ents, str):
                    ents_list = [ents]
                elif isinstance(ents, list):
                    ents_list = [str(e) for e in ents]
                # Apply filter if a subset was selected
                if sel and not (set(ents_list) & sel):
                    continue
                filt.append(r)

            if not filt:
                st.info("No YAML recognizers match the current entity selection.")
                return

            # Compact header
            st.caption(f"Showing {len(filt)} YAML recognizer entries from {file_path.name}")
            for idx, r in enumerate(filt, 1):
                name = r.get('name', f'Recognizer {idx}')
                ent = r.get('supported_entity') or r.get('supported_entities') or 'Unknown'
                lang = r.get('supported_language', 'en')
                score = r.get('score', '')
                rtype = r.get('type', 'PatternRecognizer')
                with st.expander(f"{idx}. {name} • {ent}"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.text(f"Type: {rtype}")
                        st.text(f"Language: {lang}")
                    with col_b:
                        st.text(f"Score: {score if score != '' else 'N/A'}")
                        if isinstance(ent, list):
                            st.text(f"Entities: {', '.join(ent)}")
                        else:
                            st.text(f"Entity: {ent}")
                    with col_c:
                        ctx = r.get('context') or []
                        if ctx:
                            st.text("Context:")
                            st.write(", ".join([str(c) for c in ctx]))
                    patterns = r.get('patterns') or []
                    if patterns:
                        st.markdown("**Patterns**")
                        for p in patterns:
                            if not isinstance(p, dict):
                                continue
                            pn = p.get('name', 'pattern')
                            rx = p.get('regex', '')
                            ps = p.get('score', '')
                            st.code(f"{pn} (score={ps if ps != '' else 'N/A'}):\n{rx}")
        except Exception as e:
            st.warning(f"Could not render YAML recognizers preview: {e}")

    def _list_available_code_recognizers(self) -> List[Dict[str, Any]]:
        """Import code recognizers and list their entities."""
        out: List[Dict[str, Any]] = []
        try:
            from uk_recognizers import create_uk_recognizers  # type: ignore
            recs = create_uk_recognizers()
            for r in recs:
                ents = getattr(r, 'supported_entities', None)
                if ents is None:
                    ent = getattr(r, 'supported_entity', None)
                    ents = [ent] if ent else []
                out.append({
                    'name': type(r).__name__,
                    'entities': ents,
                    'type': 'code',
                })
        except Exception:
            pass
        return out

    def _summarize_registry(self, analyzer) -> List[Dict[str, Any]]:
        """Return a simple summary of recognizers currently registered in an analyzer."""
        if not analyzer:
            return []
        try:
            reg = getattr(analyzer, 'registry', None)
            recs = getattr(reg, 'recognizers', []) if reg else []
            rows: List[Dict[str, Any]] = []
            for r in recs:
                ents = getattr(r, 'supported_entities', None)
                # Handle single-entity attribute name
                if ents is None:
                    ent = getattr(r, 'supported_entity', None)
                    ents = [ent] if ent else []
                # Try to infer the source
                cls = type(r).__name__
                source = 'built-in'
                if 'Transformer' in cls or 'transformer' in cls.lower():
                    source = 'transformer'
                elif cls == 'PatternRecognizer':
                    # Likely YAML or code recognizer
                    # Heuristic: presence of context/patterns can’t distinguish; mark as pattern
                    source = 'pattern'
                rows.append({
                    'class': cls,
                    'entities': ', '.join(ents) if isinstance(ents, list) else str(ents),
                    'source': source,
                })
            return rows
        except Exception:
            return []
    
    def save_yaml_file(self, file_path: Path, data: Dict) -> bool:
        """Save YAML file with error handling"""
        try:
            # Create timestamped backup if original exists
            if file_path.exists():
                ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                backup_name = f"{file_path.name}.bak_{ts}"
                shutil.copy2(file_path, self.backups_path / backup_name)
                # Optional: prune older than 15 backups per file
                backups = sorted([p for p in self.backups_path.glob(f"{file_path.name}.bak_*")], key=lambda p: p.stat().st_mtime, reverse=True)
                for old in backups[15:]:
                    try:
                        old.unlink()
                    except Exception:
                        pass
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving {file_path.name}: {e}")
            return False

    def list_yaml_backups(self, file_path: Path) -> List[Path]:
        """List available backups for a given YAML file (newest first)"""
        pattern = f"{file_path.name}.bak_*"
        backups = [p for p in self.backups_path.glob(pattern)]
        backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return backups

    def restore_yaml_backup(self, file_path: Path, backup_file: Path) -> bool:
        """Restore a YAML file from a selected backup"""
        try:
            if not backup_file.exists():
                st.error("Selected backup does not exist")
                return False
            shutil.copy2(backup_file, file_path)
            st.success(f"Restored {file_path.name} from {backup_file.name}")
            return True
        except Exception as e:
            st.error(f"Failed to restore backup: {e}")
            return False
    
    def get_json_results_files(self) -> List[Path]:
        """Get list of JSON result files"""
        pattern = "pii_detection_results_*.json"
        json_files = list(self.results_path.glob(pattern))
        # Sort by modification time (newest first)
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return json_files

    def get_csv_results_files(self) -> List[Path]:
        """Get list of CSV batch result files"""
        pattern = "pii_detection_results_*.csv"
        csv_files = list(self.results_path.glob(pattern))
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return csv_files
    
    def load_json_results(self, file_path: Path) -> Optional[Dict]:
        """Load JSON results file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading {file_path.name}: {e}")
            return None
    
    def run_analysis_sync(self, params: Dict) -> Dict:
        """Run PII analysis synchronously and return result"""
        try:
            cmd = [
                sys.executable,
                "test_all_recognizers.py",
                "--records", str(params['max_records']),
                "--workers", str(params['num_threads'])
            ]
            
            if params.get('use_analyzer_config'):
                cmd.extend(["--analyzer-config", "analyzers_config.yaml"])
            
            if params.get('target_field'):
                cmd.extend(["--target-field", params['target_field']])
            
            if params.get('input_file'):
                # Use full path to file in test_datasets folder
                full_input_path = self.datasets_path / params['input_file']
                cmd.extend(["--input-file", str(full_input_path)])
            
            # Explicitly control Ollama usage
            if params.get('use_ollama'):
                cmd.append("--use-ollama")
            else:
                cmd.append("--no-ollama")
            
            if params.get('builtin_only'):
                cmd.append("--builtin-only")
            
            if params.get('sequential'):
                cmd.append("--sequential")
            
            if params.get('verbose'):
                cmd.append("--verbose")
            
            # Add output directory for results
            cmd.extend(["--output-dir", str(self.results_path)])
            
            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.base_path
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'returncode': -1
            }
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("🔍 PII Control Center")
        

        # Add the PII Exposure Report page to the sidebar
        pages = [
            "🔍 Interactive Search",
            "🖼️ Slides",
            "📊 PII Exposure Report",
            "📋 System Status",
            "📖 Documentation",
        ]

        # Respect current page from session to allow programmatic navigation
        current = st.session_state.get('__current_page__', pages[0])
        try:
            default_index = pages.index(current)
        except ValueError:
            default_index = 0

        page = st.sidebar.selectbox(
            "Choose Function",
            options=pages,
            index=default_index,
            key='__page_select__'
        )
        st.session_state['__current_page__'] = page
        return page

    def render_exposure_report_page(self):
        """Render the PII Exposure Report page"""
        if not REPORT_GENERATOR_AVAILABLE:
            st.error("The PII Exposure Report module could not be imported. Please ensure report_generator.py is present.")
            return
        # Use the results_path for batch results
        display_exposure_report(self.results_path)
    
    def render_analysis_page(self):
        """Render the analysis execution page"""
        st.title("🚀 Run PII Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Analysis Configuration")
            
            # Basic settings
            max_records = st.number_input(
                "Maximum Records to Process",
                min_value=1,
                max_value=10000,
                value=100,
                help="Number of records to analyze"
            )
            
            num_threads = st.slider(
                "Number of Threads",
                min_value=1,
                max_value=8,
                value=4,
                help="Parallel processing threads"
            )
            
            # File selection - look in test_datasets folder
            csv_files = list(self.datasets_path.glob("*.csv"))
            if csv_files:
                file_names = [f.name for f in csv_files]
                options = [""] + file_names
                # Prefer the cleaned dataset if available
                preferred = "unified_test_analyzer_format_clean.csv"
                fallback = "unified_test_analyzer_format.csv"
                default_index = 0
                if preferred in file_names:
                    default_index = options.index(preferred)
                elif fallback in file_names:
                    default_index = options.index(fallback)

                input_file = st.selectbox(
                    "Input Dataset",
                    options=options,
                    index=default_index,
                    help="Choose CSV file to analyze from test_datasets folder"
                )
            else:
                st.warning("No CSV files found in the test_datasets directory")
                input_file = ""
            
            # Target field
            target_field = st.selectbox(
                "Target Field",
                options=["source", "customer_message", "agent_response", "corrupted"],
                index=0,
                help="Field to analyze for PII"
            )
            
            # Configuration options
            st.subheader("Configuration Options")
            
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                use_analyzer_config = st.checkbox(
                    "Use YAML Analyzer Config",
                    value=True,
                    help="Use analyzers_config.yaml for analyzer settings"
                )
                
                sequential = st.checkbox(
                    "Sequential Processing",
                    value=False,
                    help="Use single-threaded processing"
                )
                
                verbose = st.checkbox(
                    "Verbose Output",
                    value=False,
                    help="Enable detailed logging"
                )
            
            with col_opt2:
                # Default Use Ollama based on analyzers_config if available
                try:
                    _an_cfg = self.load_yaml_file(self.analyzer_config_path)
                    _ollama_default = bool(_an_cfg.get('ollama', {}).get('enabled', False))
                except Exception:
                    _ollama_default = False

                use_ollama = st.checkbox(
                    "Use Ollama",
                    value=_ollama_default,
                    help="Include local LLM analyzer (Ollama) in batch runs"
                )
                
                builtin_only = st.checkbox(
                    "Built-in Presidio Only",
                    value=False,
                    help="Use ONLY built-in Presidio recognizers (no custom UK, pattern, or Ollama analyzers)"
                )
        
        with col2:
            st.header("Quick Actions")
            
            # Run analysis button
            if not st.session_state.analysis_running:
                if st.button("🚀 Run Analysis", type="primary", width="stretch"):
                    params = {
                        'max_records': max_records,
                        'num_threads': num_threads,
                        'input_file': input_file if input_file else None,
                        'target_field': target_field,
                        'use_analyzer_config': use_analyzer_config,
                        'sequential': sequential,
                        'verbose': verbose,
                        'use_ollama': use_ollama,
                        'builtin_only': builtin_only
                    }
                    
                    # Store params and start analysis
                    st.session_state.analysis_params = params
                    st.session_state.analysis_start_time = datetime.now()
                    st.session_state.analysis_running = True
                    st.rerun()
                    
            else:
                # Show running analysis
                start_time = st.session_state.analysis_start_time
                if start_time:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    st.info(f"🔄 Analysis running for {elapsed:.0f} seconds...")
                else:
                    st.info("🔄 Analysis starting...")
                
                if st.button("🛑 Stop Analysis", width="stretch"):
                    st.session_state.analysis_running = False
                    st.session_state.analysis_start_time = None
                    st.rerun()
            
            # Quick status
            st.subheader("System Status")
            
            # Check if analyzer config exists
            if self.analyzer_config_path.exists():
                st.success("✅ Analyzer config ready")
            else:
                st.error("❌ Analyzer config missing")
            
            # Check for recent results
            json_files = self.get_json_results_files()
            if json_files:
                latest_file = json_files[0]
                mod_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
                st.info(f"📊 Latest results: {mod_time.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.warning("⚠️ No results found")
        
        # Handle running analysis
        if st.session_state.analysis_running and st.session_state.analysis_params:
            st.info("🔄 Analysis in progress...")
            
            # Create a progress container
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run analysis synchronously
                with st.spinner("Running PII analysis..."):
                    result = self.run_analysis_sync(st.session_state.analysis_params)
                
                # Update progress to completion
                progress_bar.progress(100)
                
                # Handle results
                st.session_state.analysis_running = False
                st.session_state.analysis_start_time = None
                st.session_state.analysis_params = None
                
                if result['success']:
                    st.success("✅ Analysis completed successfully!")
                    
                    # Try to load the most recent results
                    json_files = self.get_json_results_files()
                    if json_files:
                        st.session_state.analysis_results = self.load_json_results(json_files[0])
                        st.info(f"📊 Results loaded from: {json_files[0].name}")
                    
                    # Show some output if verbose
                    if result.get('stdout'):
                        with st.expander("Analysis Output"):
                            st.text(result['stdout'][-2000:])  # Show last 2000 chars
                else:
                    st.error("❌ Analysis failed!")
                    if result.get('stderr'):
                        st.error(f"Error: {result['stderr']}")
                    if result.get('error'):
                        st.error(f"Exception: {result['error']}")
                
                # Auto-refresh to clear the progress
                time.sleep(1)
                st.rerun()
    
    def render_interactive_search_page(self):
        """Render interactive PII search page"""
        st.title("🔍 Interactive PII Search")
        st.markdown("Enter text to search for PII entities using your current configuration.")

        # Quick access to the PII Exposure Report
        with st.container():
            c1, c2 = st.columns([5, 1])
            with c2:
                if st.button("📊 Open PII Exposure Report", help="View report generated from batch results"):
                    st.session_state['__current_page__'] = "📊 PII Exposure Report"
                    _st_rerun()
        
        # Load example snippets from config.yaml
        examples: List[str] = []
        try:
            cfg = self.load_yaml_file(self.config_path)
            if isinstance(cfg, dict):
                ex = cfg.get('interactive_examples', [])
                if isinstance(ex, list):
                    examples = [str(x) for x in ex if isinstance(x, (str, int, float))]
        except Exception:
            examples = []

        # Example selector
        options = examples + ["Custom input"] if examples else ["Custom input"]
        selected_example = st.selectbox(
            "Examples",
            options=options,
            index=0,
            help="Choose a sample text from config.yaml or switch to custom input"
        )

        # Determine default text for the editor
        default_text = ""
        if examples:
            default_text = examples[0]
        if selected_example != "Custom input" and selected_example:
            default_text = selected_example

        # Text input area (editable)
        search_text = st.text_area(
            "Enter text to search for PII:",
            value=default_text,
            placeholder="Example: My name is John Smith and my email is john.smith@company.com. I was born on 15/08/1990.",
            height=150
        )

        # Save current text as a reusable example
        save_col, _ = st.columns([1, 5])
        with save_col:
            if st.button("💾 Save as Example", help="Add the current text to interactive_examples in config.yaml"):
                if search_text and search_text.strip():
                    try:
                        cfg2 = self.load_yaml_file(self.config_path) if self.config_path.exists() else {}
                        if not isinstance(cfg2, dict):
                            cfg2 = {}
                        current = cfg2.get('interactive_examples', [])
                        # Deduplicate while preserving order
                        new_list = []
                        seen = set()
                        for item in ([search_text] + (current if isinstance(current, list) else [])):
                            if isinstance(item, (str, int, float)):
                                s = str(item)
                                if s not in seen:
                                    seen.add(s)
                                    new_list.append(s)
                        # Keep most recent first, cap to 25 items
                        cfg2['interactive_examples'] = new_list[:25]
                        if self.save_yaml_file(self.config_path, cfg2):
                            st.success("✅ Saved example to config.yaml")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to save example: {e}")
        
        # ================= New Simplified Configuration UI =================
        st.markdown("### ⚙️ Configuration")
        c1, c2, c3 = st.columns(3)

        # Column 1: High-level engine toggles
        with c1:
            built_in_only = st.checkbox("Built-in Presidio Only", value=False, help="Disable transformers, YAML patterns, custom code recognizers, and Ollama")
            use_transformers = False if built_in_only else st.checkbox("Use Transformers", value=True, help="Include transformer-based entity recognition")
            transformer_models_sel: List[str] = []
            if use_transformers and not built_in_only:
                with st.expander("🤗 Transformer Models", expanded=False):
                    default_models = [
                        "dslim/bert-base-NER",
                        "elastic/distilbert-base-cased-finetuned-conll03-english",
                        "Jean-Baptiste/roberta-large-ner-english",
                        "dbmdz/bert-large-cased-finetuned-conll03-english",
                    ]
                    transformer_models_sel = st.multiselect(
                        "Select models",
                        options=default_models,
                        default=["dslim/bert-base-NER"],
                        help="Pick multiple HF NER models"
                    )
            # Ollama toggle disabled in built-in only mode
            try:
                _an_cfg2 = self.load_yaml_file(self.analyzer_config_path)
                _ollama_default_interactive = bool(_an_cfg2.get('ollama', {}).get('enabled', False))
            except Exception:
                _ollama_default_interactive = False
            use_ollama = False if built_in_only else st.checkbox("Use Ollama", value=_ollama_default_interactive, help="Include local LLM analysis")

        # Column 2: Pattern/YAML & code recognizers
        with c2:
            use_yaml_patterns = False if built_in_only else st.checkbox("Use YAML Pattern Recognizers", value=True, help="Enable recognizers defined in presidio_recognizers_config.yaml")
            use_code_recognizers = False if built_in_only else st.checkbox("Use Custom Code Recognizers", value=True, help="Enable project-specific custom recognizers (e.g., UK_* variants)")
            # Specific override: replace built-in DateRecognizer with YAML Date/Time
            yaml_datetime_override = False if built_in_only else st.checkbox(
                "Use YAML Date/Time (replace built-in)",
                value=True,
                help="Removes Presidio's built-in DateRecognizer and uses the YAML Date/Time recognizer to reduce false positives (e.g., numbers misread as dates)."
            )

            # Choose which YAML file to use for recognizers when enabled
            selected_yaml_path: Optional[Path] = None
            yaml_file_label = "presidio_recognizers_config.yaml"
            if use_yaml_patterns and not built_in_only:
                candidates = self._find_yaml_candidates()
                # Default to session choice or project default
                default_yaml = st.session_state.get('__yaml_recognizers_path__', str(self.recognizers_config_path))
                # Make a mapping for display
                options = [str(p) for p in candidates] or [str(self.recognizers_config_path)]
                try:
                    default_index = options.index(default_yaml) if default_yaml in options else 0
                except Exception:
                    default_index = 0
                chosen = st.selectbox(
                    "Recognizers YAML file",
                    options=options,
                    index=default_index,
                    help="Pick which YAML file to load no-code recognizers from"
                )
                # Allow manual override
                manual_path = st.text_input("Or enter a custom YAML path", value="", help="Absolute or relative path to a YAML/YML file")
                use_path_str = manual_path.strip() or chosen
                try:
                    selected_yaml_path = Path(use_path_str)
                    yaml_file_label = os.path.relpath(str(selected_yaml_path), str(self.base_path)) if selected_yaml_path.is_absolute() else str(selected_yaml_path)
                    # Persist for the session
                    st.session_state['__yaml_recognizers_path__'] = str(selected_yaml_path)
                except Exception:
                    selected_yaml_path = self.recognizers_config_path
                    st.session_state['__yaml_recognizers_path__'] = str(self.recognizers_config_path)

            yaml_entities: List[str] = []
            if use_yaml_patterns and not built_in_only:
                try:
                    # Use selected YAML file if provided
                    ypath = selected_yaml_path if selected_yaml_path else Path("presidio_recognizers_config.yaml")
                    if ypath.exists():
                        ycfg = self.load_yaml_file(ypath)
                        if isinstance(ycfg, dict):
                            recs = ycfg.get('recognizers', [])
                            for r in recs:
                                if isinstance(r, dict):
                                    ents = r.get('supported_entity') or r.get('supported_entities')
                                    if isinstance(ents, str):
                                        yaml_entities.append(ents)
                                    elif isinstance(ents, list):
                                        yaml_entities.extend([str(e) for e in ents])
                    yaml_entities = sorted(set(yaml_entities))
                except Exception:
                    yaml_entities = []
            selected_yaml_entities: List[str] = []
            if use_yaml_patterns and yaml_entities:
                selected_yaml_entities = st.multiselect("YAML Entities", options=yaml_entities, default=yaml_entities, help="Limit which YAML-based entities are active")

            # Detailed preview of YAML recognizers that will be loaded
            if use_yaml_patterns and not built_in_only and (selected_yaml_path or self.recognizers_config_path.exists()):
                with st.expander("📄 YAML recognizers to be loaded", expanded=False):
                    preview_path = selected_yaml_path if selected_yaml_path else self.recognizers_config_path
                    self._render_yaml_entries_preview(preview_path, selected_yaml_entities)

            code_entities_master = [
                "UK_POSTCODE","UK_SORT_CODE","UK_NHS_NUMBER","UK_NI_NUMBER","UK_BANK_ACCOUNT","UK_MONETARY_VALUE",
                "UK_PASSPORT","CUSTOMER_ID","UK_PHONE_NUMBER","ENHANCED_EMAIL","ENHANCED_CREDIT_CARD","ENHANCED_IBAN",
                "UK_VAT_NUMBER","UK_COMPANY_NUMBER","UK_VEHICLE_REG","UK_DRIVING_LICENSE","URL","IP_ADDRESS"
            ]
            selected_code_entities: List[str] = []
            if use_code_recognizers:
                selected_code_entities = st.multiselect("Code Recognizers", options=code_entities_master, default=code_entities_master, help="Choose custom code recognizers to include")

            with st.expander("🔎 Recognizers overview", expanded=False):
                # Preview lists (static from files/modules) based on toggles
                avail_yaml = self._list_available_yaml_recognizers() if use_yaml_patterns and not built_in_only else []
                avail_code = self._list_available_code_recognizers() if use_code_recognizers and not built_in_only else []
                st.write("Available (by source):")
                c_av1, c_av2 = st.columns(2)
                with c_av1:
                    st.markdown(f"• Built-in Presidio: enabled")
                    st.markdown(f"• YAML recognizers: {'enabled' if avail_yaml else 'disabled or none'}")
                    st.markdown(f"• Code recognizers: {'enabled' if avail_code else 'disabled or none'}")
                with c_av2:
                    if avail_yaml:
                        st.caption("YAML entities")
                        st.write(sorted({e for r in avail_yaml for e in r.get('entities', [])}))
                    if avail_code:
                        st.caption("Code entities")
                        st.write(sorted({e for r in avail_code for e in r.get('entities', [])}))
                st.info("Actual 'used recognizers' will be listed after you run analysis, reflecting these toggles.")

        # Column 3: Entity filtering & thresholds
        with c3:
            sensitivity = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.35, 0.05, help="Lower values detect more entities")
            max_results = st.number_input("Max Results", 1, 50, 20, help="Maximum entities to display")
            selected_entities = st.multiselect(
                "Entity Filter (optional)",
                ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "DATE_TIME", "LOCATION", "CREDIT_CARD", "UK_POSTCODE", "UK_NHS_NUMBER"],
                help="If set, only these entity types will be shown"
            )
            with st.expander("🔧 Built-in Presidio recognizers control", expanded=False):
                st.caption("Enable/disable builtin entity types (applies before YAML/code recognizers). Leave both empty for default behavior.")
                builtin_all = [
                    "PERSON","PHONE_NUMBER","EMAIL_ADDRESS","DOMAIN_NAME","IBAN_CODE","CREDIT_CARD","CRYPTO","DATE_TIME","NRP","LOCATION","MEDICAL_LICENSE",
                    "URL","IP_ADDRESS","UK_NHS_NUMBER","UK_NI_NUMBER","US_SSN","US_DRIVER_LICENSE","US_ITIN","US_PASSPORT","UK_PASSPORT"
                ]
                disable_builtin_entities = st.multiselect(
                    "Disable these built-in entities",
                    options=builtin_all,
                    default=[],
                    help="Selected entities will be removed from Presidio built-ins"
                )
                enable_only_builtin_entities = st.multiselect(
                    "Enable ONLY these built-in entities",
                    options=builtin_all,
                    default=[],
                    help="If set, Presidio built-ins will be restricted to this set (after which disabled list is applied)"
                )
                # Persist to session for both runs
                st.session_state['__disable_builtin_entities__'] = disable_builtin_entities
                st.session_state['__enable_only_builtin_entities__'] = enable_only_builtin_entities

        st.markdown("---")
        # Expose Presidio disable option BEFORE run so user can toggle
        disable_presidio_default = st.session_state.get('__disable_presidio__', False)
        st.session_state['__disable_presidio__'] = st.checkbox(
            "Disable built-in Presidio engine (use only Transformers / YAML / Code / Ollama)",
            value=disable_presidio_default,
            help="When enabled, Presidio built-in recognizers are not loaded."
        )

    # Output mode selection removed; we will always show both Anonymized and Tokenized outputs

        run_disabled = not search_text.strip()
        if st.button("🚀 Run Analysis", type="primary", disabled=run_disabled):
            if search_text.strip():
                # Derive effective flags
                eff_use_transformers = use_transformers and not built_in_only
                eff_use_ollama = use_ollama and not built_in_only
                # Store selection metadata for downstream filtering
                st.session_state['__yaml_entity_subset__'] = selected_yaml_entities
                st.session_state['__code_entity_subset__'] = selected_code_entities
                # If YAML Date/Time override is on, force YAML patterns on so the YAML Date recognizer loads
                st.session_state['__use_yaml_patterns__'] = (use_yaml_patterns or yaml_datetime_override) and not built_in_only
                st.session_state['__use_code_recognizers__'] = use_code_recognizers and not built_in_only
                st.session_state['__built_in_only__'] = built_in_only
                # disable flag already set above
                try:
                    tm_sel = transformer_models_sel if eff_use_transformers else []
                except NameError:
                    tm_sel = []
                # Always anonymize and run both analyses; custom DATE_TIME removed per request
                anonymize_results = True
                # Use YAML Date/Time override to drive removal of built-in DateRecognizer
                use_custom_datetime = bool(yaml_datetime_override)
                self._perform_interactive_search(
                    search_text, eff_use_transformers, eff_use_ollama, sensitivity,
                    selected_entities, anonymize_results, max_results, use_custom_datetime, tm_sel
                )
                # Enhanced recognizer analysis automatically after primary search
                self._perform_enhanced_analysis(
                    search_text, eff_use_transformers, eff_use_ollama, sensitivity,
                    selected_entities, anonymize_results, max_results, use_custom_datetime, tm_sel
                )

        payload = st.session_state.get('__interactive_last_payload__')
        if payload:
            self._render_interactive_output(payload)

        st.markdown("---")
        self._render_interactive_batch_section()

        st.markdown("---")
        self._render_inline_results_and_exposure_section()

    def _render_interactive_batch_section(self) -> None:
        """Render an inline batch processing section within Interactive Search.
        Lets the user pick a dataset, choose columns and entity types, and run batch analysis
        using the current engine configuration. Results can be previewed and downloaded.
        """
        st.subheader("⚡ Batch processing (interactive)")
        st.caption("Runs batch analysis with the same engine settings above. Pick a dataset and target columns.")

        # Discover CSV datasets in the configured datasets folder
        csv_files = sorted([p for p in self.datasets_path.glob("*.csv")])
        up_col, sel_col = st.columns([1, 2])
        with up_col:
            uploaded = st.file_uploader("Upload CSV to test_datasets", type=["csv"], help="Uploads are saved under test_datasets/ for reuse")
            if uploaded is not None:
                try:
                    dst = self.datasets_path / uploaded.name
                    with open(dst, 'wb') as f:
                        f.write(uploaded.getbuffer())
                    st.success(f"Saved: {dst.name}")
                    # Refresh list
                    csv_files = sorted([p for p in self.datasets_path.glob("*.csv")])
                except Exception as e:
                    st.error(f"Failed to save upload: {e}")
        with sel_col:
            if csv_files:
                file_labels = [p.name for p in csv_files]
                # Prefer the cleaned dataset if available, else the original unified file
                preferred = "unified_test_analyzer_format_clean.csv"
                fallback = "unified_test_analyzer_format.csv"
                try:
                    default_index = file_labels.index(preferred)
                except ValueError:
                    try:
                        default_index = file_labels.index(fallback)
                    except ValueError:
                        default_index = 0

                sel_idx = st.selectbox(
                    "Data source (CSV)",
                    options=list(range(len(csv_files))),
                    index=default_index,
                    format_func=lambda i: file_labels[i]
                )
                selected_csv = csv_files[sel_idx]
            else:
                selected_csv = None
                st.info("No CSV files found in test_datasets/. Upload one to begin.")

        if not selected_csv:
            return

        # Read a small sample to infer columns
        import pandas as pd
        try:
            df_head = pd.read_csv(selected_csv, nrows=200)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

        all_columns = list(df_head.columns)
        text_like_cols = [c for c in all_columns if str(df_head[c].dtype) == 'object'] or all_columns

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            # Default to 'source' column when present
            default_text_cols = ["source"] if "source" in all_columns else (
                text_like_cols[:3] if len(text_like_cols) >= 3 else text_like_cols
            )
            selected_columns = st.multiselect(
                "Text columns to analyze",
                options=all_columns,
                default=default_text_cols,
                help="These columns' text will be scanned for PII"
            )
        with c2:
            # Offer a pragmatic set of entities (built-in + UK-focused)
            entity_options = [
                "PERSON","EMAIL_ADDRESS","PHONE_NUMBER","DATE_TIME","LOCATION","CREDIT_CARD","IBAN_CODE","IP_ADDRESS","URL","ORGANIZATION",
                "US_SSN","US_DRIVER_LICENSE","US_PASSPORT",
                "UK_POSTCODE","UK_SORT_CODE","UK_NI_NUMBER","UK_NHS_NUMBER","UK_BANK_ACCOUNT","UK_MONETARY_VALUE","UK_PASSPORT"
            ]
            selected_entities = st.multiselect(
                "Entities to detect",
                options=entity_options,
                default=["PERSON","EMAIL_ADDRESS","PHONE_NUMBER","CREDIT_CARD","UK_POSTCODE"],
                help="Only these entity types will be kept in the batch results"
            )
        with c3:
            max_records = st.number_input("Max rows", min_value=1, max_value=200000, value=min(1000, len(df_head)), step=50)
            sensitivity = st.slider("Sensitivity", 0.0, 1.0, 0.35, 0.05, help="Lower detects more")

        # Extra fields to include in export (non-analyzed columns)
        extra_cols = st.multiselect(
            "Extra columns to include in output (optional)",
            options=[c for c in all_columns if c not in selected_columns],
            help="These columns will be carried forward into the exported results"
        )

        # Option to automatically save JSON results into results/ for immediate viewing
        save_to_results = st.checkbox(
            "Save JSON to results/ after run",
            value=True,
            help="If enabled, the JSON output will be written into results/ and appear in the results picker below."
        )

        run_disabled = not selected_columns or not selected_entities
        if st.button("▶️ Run interactive batch", disabled=run_disabled):
            self._run_interactive_batch(
                csv_path=selected_csv,
                columns=selected_columns,
                entities=selected_entities,
                extra_columns=extra_cols,
                max_rows=int(max_records),
                sensitivity=float(sensitivity),
                save_to_results=bool(save_to_results),
            )

    def _run_interactive_batch(self, csv_path, columns, entities, extra_columns, max_rows, sensitivity: float, save_to_results: bool = False):
        """Execute batch analysis with current engine configuration and render results & exports."""
        import pandas as pd
        from pii_analysis_cli import load_presidio_engines, enhanced_pii_analysis
        from io import StringIO

        # Build config from current session settings to mirror Interactive Search
        config: Dict[str, Any] = {}
        config['sensitivity'] = sensitivity
        disable_presidio = bool(st.session_state.get('__disable_presidio__', False))
        config['use_presidio'] = not disable_presidio
        built_in_only = bool(st.session_state.get('__built_in_only__', False))
        if built_in_only:
            config['use_uk_patterns'] = False
            config['__use_yaml_patterns__'] = False
            config['__use_code_recognizers__'] = False
            config['__built_in_only__'] = True
        else:
            config['use_uk_patterns'] = True
            config['__use_yaml_patterns__'] = st.session_state.get('__use_yaml_patterns__', True)
            config['__use_code_recognizers__'] = st.session_state.get('__use_code_recognizers__', True)
            config['__yaml_entity_subset__'] = st.session_state.get('__yaml_entity_subset__', [])
            config['__code_entity_subset__'] = st.session_state.get('__code_entity_subset__', [])
            y_path = st.session_state.get('__yaml_recognizers_path__')
            if y_path:
                config['__yaml_recognizers_path__'] = y_path
        # Built-in entity allow/deny lists
        dis_bi = st.session_state.get('__disable_builtin_entities__', [])
        en_only_bi = st.session_state.get('__enable_only_builtin_entities__', [])
        if dis_bi:
            config['__disable_builtin_entities__'] = dis_bi
        if en_only_bi:
            config['__enable_only_builtin_entities__'] = en_only_bi

        # Initialize engines once
        try:
            analyzer, anonymizer = load_presidio_engines(config)
        except Exception as e:
            analyzer = None
            anonymizer = None
            st.warning(f"Proceeding without Presidio anonymizer due to init error: {e}")

        # Load data
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

        total = min(len(df), max_rows)
        if total <= 0:
            st.info("No rows to process.")
            return

        st.info(f"Processing {total:,} rows from {os.path.basename(str(csv_path))}…")
        progress = st.progress(0)
        status = st.empty()

        rows_out: List[Dict[str, Any]] = []
        # Detailed per-record aggregation: {row_index: { 'columns': {col: {...}}, 'extra': {...} }}
        record_details: Dict[int, Dict[str, Any]] = {}
        entity_counts: Dict[str, int] = {}
        # Track aggregated PII context per row for downstream exposure tooling
        row_pii_summary: Dict[int, Dict[str, Any]] = {}
        dataset_name = os.path.basename(str(csv_path))
        start_time = time.time()

        # Iterate rows
        for i, (_, row) in enumerate(df.head(total).iterrows(), start=1):
            for col in columns:
                val = row.get(col, None)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                text = str(val)
                if not text.strip():
                    continue

                # Run analysis limited to selected entities
                res = enhanced_pii_analysis(text, analyzer, config=config, selected_entities=entities)
                findings = [f for f in (res.get('all_findings') or []) if f.get('entity_type') in set(entities)]

                # Anonymize only if there are findings, else keep original text
                if findings:
                    anon_text = self._get_anonymized_text(text, findings, config)
                else:
                    anon_text = text

                # Compute tokenized output (format-preserving) from detections
                try:
                    tokenized_text = self._tokenize_from_detections(text, findings)
                except Exception:
                    # Fallback to masking/tokenization if tokenizer fails
                    tokenized_text = self._fallback_anonymize_from_detections(text, findings)

                # Aggregate per-record details for detailed view
                r_index = i - 1
                if r_index not in record_details:
                    record_details[r_index] = {
                        'columns': {},
                        'extra': ({k: row.get(k) for k in extra_columns} if extra_columns else {})
                    }
                record_details[r_index]['columns'][col] = {
                    'original': text,
                    'anonymized': anon_text,
                    'tokenized': tokenized_text,
                    'entities': findings,
                }

                if findings:
                    summary = row_pii_summary.setdefault(r_index, {
                        'fields': set(),
                        'values': []
                    })
                    fields_set: Set[str] = summary['fields']  # type: ignore[assignment]
                    values_list: List[str] = summary['values']  # type: ignore[assignment]
                    for f in findings:
                        ent_type = f.get('entity_type')
                        if ent_type:
                            fields_set.add(str(ent_type))
                        raw_val = f.get('text')
                        if raw_val:
                            raw_str = str(raw_val)
                            if raw_str not in values_list:
                                values_list.append(raw_str)
                    record_details[r_index]['pii_fields_used'] = sorted(fields_set)
                    record_details[r_index]['pii_raw_values'] = values_list.copy()

                # Tally counts and prepare flattened rows
                for f in findings:
                    et = f.get('entity_type')
                    entity_counts[et] = entity_counts.get(et, 0) + 1
                    summary = row_pii_summary.get(r_index, {'fields': set(), 'values': []})
                    fields_used = sorted(list(summary.get('fields', [])))
                    raw_values = list(summary.get('values', []))
                    rows_out.append({
                        'row_index': i-1,
                        'column': col,
                        'source': col,
                        'source_dataset': dataset_name,
                        'entity_type': et,
                        'text': f.get('text'),
                        'start': f.get('start'),
                        'end': f.get('end'),
                        'confidence': f.get('confidence'),
                        'recognizer': f.get('recognizer') or f.get('source') or '',
                        # Per-text analyzer timings (seconds) from the current analysis call
                        'analyzer_timings': res.get('timings', {}),
                        'pii_fields_used': fields_used,
                        'pii_raw_values': raw_values,
                        # Include full field variants for exposure checks
                        'original_column_text': text,
                        'tokenized_column_text': tokenized_text,
                        'anonymized_column_text': anon_text,
                        **({k: row.get(k) for k in extra_columns} if extra_columns else {})
                    })

            # Update progress
            progress.progress(min(i / total, 1.0))
            if i % 25 == 0 or i == total:
                status.text(f"Processed {i:,}/{total:,} rows…")

        elapsed = time.time() - start_time
        st.success(f"Done. {len(rows_out):,} findings across {total:,} rows in {elapsed:.1f}s")

        # Show small summary and preview
        if entity_counts:
            st.write({k: v for k, v in sorted(entity_counts.items(), key=lambda kv: (-kv[1], kv[0]))})
        if rows_out:
            df_out = pd.DataFrame(rows_out)
            st.dataframe(df_out.head(200), width="stretch")

            # Exports
            include_full_fields = st.checkbox(
                "Include original/tokenized/anonymized columns in CSV export",
                value=True,
                key="batch_csv_include_full_fields",
                help="Uncheck to drop original_column_text, tokenized_column_text, and anonymized_column_text from the CSV file. JSON export always includes full context."
            )
            if include_full_fields:
                df_csv = df_out
            else:
                drop_cols = [
                    'original_column_text',
                    'tokenized_column_text',
                    'anonymized_column_text',
                ]
                df_csv = df_out.drop(columns=[c for c in drop_cols if c in df_out.columns])

            csv_buf = StringIO()
            df_csv.to_csv(csv_buf, index=False)
            st.download_button(
                "💾 Download CSV",
                data=csv_buf.getvalue(),
                file_name=f"interactive_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            json_data = {
                'metadata': {
                    'source': os.path.basename(str(csv_path)),
                    'rows_processed': int(total),
                    'entities': list(entities),
                    'pii_fields_detected': sorted({
                        str(field)
                        for summary in row_pii_summary.values()
                        for field in summary.get('fields', set())
                    }),
                    'pii_raw_value_count': int(sum(len(summary.get('values', [])) for summary in row_pii_summary.values())),
                    'columns': list(columns),
                    'extra_columns': list(extra_columns) if extra_columns else [],
                    'sensitivity': float(sensitivity),
                    'use_presidio': bool(config.get('use_presidio', True)),
                    'built_in_only': bool(st.session_state.get('__built_in_only__', False)),
                    'generated_at': datetime.now().isoformat(),
                },
                'results': rows_out,
            }
            st.download_button(
                "💾 Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"interactive_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

            with st.expander("📄 Preview CSV output", expanded=False):
                preview_rows = min(len(df_csv), 500)
                st.dataframe(df_csv.head(preview_rows), width="stretch")

            with st.expander("🧾 Preview JSON output", expanded=False):
                st.json(json_data)

            # Optionally persist to results/ so it shows up in the inline viewer immediately
            if save_to_results:
                try:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    base_name = f"pii_detection_results_interactive_{ts}"
                    json_name = f"{base_name}.json"
                    csv_name = f"{base_name}.csv"
                    json_path = self.results_path / json_name
                    csv_path = self.results_path / csv_name
                    df_csv.to_csv(csv_path, index=False)
                    with open(json_path, 'w') as f:
                        json.dump(json_data, f, indent=2)
                    st.success(f"Saved to results/: {csv_name}, {json_name}")
                    # Remember JSON file as the last-selected results file
                    st.session_state['__last_results_filename__'] = json_name
                except Exception as e:
                    st.warning(f"Could not save to results/: {e}")

            # Detailed per-record preview section
            with st.expander("Detailed per-record preview (Original | Tokenized | Anonymized + entities)", expanded=False):
                if not record_details:
                    st.info("No detailed records to display.")
                else:
                    # Controls
                    only_with_entities = st.checkbox("Show only rows with detected entities", value=True, key="batch_detail_only_entities")
                    max_preview = st.slider("Max rows to preview", min_value=1, max_value=max(5, min(100, len(record_details))), value=min(20, len(record_details)))
                    # Use the full analyzer logic for exposure classification
                    exposure_analyzer = PIIExposureAnalyzer()

                    def _rank(r: str) -> int:
                        return {"NO": 0, "PARTIAL": 1, "FULL": 2}.get(r, 0)

                    def _compute_field_exposure(original_text: str, anonymized_text: str, entities_list: List[Dict[str, Any]]):
                        """Compute exposure per field using PIIExposureAnalyzer on each entity text.
                        Returns: (level: NO|PARTIAL|FULL, details: List[Dict]) where details contains per-entity exposure info.
                        """
                        if not anonymized_text or not original_text or not entities_list:
                            return "NO", []
                        worst = "NO"
                        details = []
                        for e in entities_list:
                            ent_text = (e.get('text') or '').strip()
                            if not ent_text:
                                continue
                            level, exposed_parts = exposure_analyzer.check_exposure_level(ent_text, anonymized_text)
                            # Map to compact labels
                            lvl = "NO" if level == 'NO_EXPOSURE' else ("FULL" if level == 'FULL_EXPOSURE' else "PARTIAL")
                            if _rank(lvl) > _rank(worst):
                                worst = lvl
                            if lvl != "NO":
                                details.append({
                                    'entity_type': e.get('entity_type') or e.get('type'),
                                    'text': ent_text,
                                    'level': lvl,
                                    'exposed_snippets': exposed_parts
                                })
                        return worst, details

                    COMMON_WORDS = {
                        'the','and','for','with','this','that','from','have','has','are','was','were','you','your','his','her','their','our','not','but','all','any','can','will','would','there','here','about','into','over','under','more','most','some','such','only','other','than','then','also','between','after','before','because','where','when','which','while','on','in','at','to','of','a','an','is','it','as','by','be'
                    }

                    def _compute_non_pii_retained(original_text: str, anonymized_text: str, entities_list: List[Dict[str, Any]]):
                        """Identify non-PII tokens from the original field that still appear in anonymized text.
                        We remove entity spans by index when available, then tokenize both sides and intersect.
                        Returns: (count, samples: List[str])
                        """
                        if not original_text or not anonymized_text:
                            return 0, []

                        # Remove entity spans from original to isolate non-PII content
                        non_pii_text = original_text
                        try:
                            if entities_list:
                                mask = [True] * len(original_text)
                                for e in entities_list:
                                    s = e.get('start')
                                    t = e.get('end')
                                    if isinstance(s, int) and isinstance(t, int) and 0 <= s < t <= len(original_text):
                                        for i in range(s, t):
                                            mask[i] = False
                                non_pii_text = ''.join(ch if mask[i] else ' ' for i, ch in enumerate(original_text))
                        except Exception:
                            # Fallback: best-effort removal by replacing entity texts
                            if entities_list:
                                for e in entities_list:
                                    et = (e.get('text') or '').strip()
                                    if et:
                                        non_pii_text = non_pii_text.replace(et, ' ')

                        orig_tokens = exposure_analyzer.tokenize_text(non_pii_text)
                        anon_tokens = exposure_analyzer.tokenize_text(anonymized_text)
                        # Filter tokens: length>=3 or digits>=4 and not common words
                        def _keep(tok: str) -> bool:
                            return ((tok.isdigit() and len(tok) >= 4) or (len(tok) >= 3 and tok.isalpha())) and tok not in COMMON_WORDS
                        retained = sorted(t for t in (orig_tokens & anon_tokens) if _keep(t))
                        return len(retained), retained[:10]

                    # Iterate rows in order
                    shown = 0
                    for r_index in sorted(record_details.keys()):
                        rec = record_details[r_index]
                        cols_data = rec.get('columns', {})
                        # Count entities across columns
                        total_entities = sum(len((cdata.get('entities') or [])) for cdata in cols_data.values())
                        if only_with_entities and total_entities == 0:
                            continue

                        # Pre-compute analyzer-based exposure per field and retained non-PII
                        row_risk = "NO"
                        for cname, cdata in cols_data.items():
                            original_text = cdata.get('original', '')
                            anonymized_text = cdata.get('anonymized', '')
                            ents = cdata.get('entities') or []
                            field_level, field_details = _compute_field_exposure(original_text, anonymized_text, ents)
                            cdata['exposure_level'] = field_level
                            cdata['exposure_details'] = field_details
                            if _rank(field_level) > _rank(row_risk):
                                row_risk = field_level
                            # Non-PII retained tokens
                            np_count, np_samples = _compute_non_pii_retained(original_text, anonymized_text, ents)
                            cdata['non_pii_retained_count'] = np_count
                            cdata['non_pii_retained_samples'] = np_samples

                        header = f"Row {r_index} — {total_entities} entit{'y' if total_entities==1 else 'ies'} — Risk: {row_risk}"
                        fields_used = rec.get('pii_fields_used') or []
                        raw_values = rec.get('pii_raw_values') or []
                        if fields_used:
                            header += f" — Fields: {', '.join(fields_used)}"
                        if raw_values:
                            header += f" — Raw: {', '.join(raw_values[:3])}{'…' if len(raw_values) > 3 else ''}"
                        if rec.get('extra'):
                            # Optionally show a compact summary of extra columns
                            extra_preview = ", ".join(f"{k}={v}" for k, v in list(rec['extra'].items())[:4])
                            if extra_preview:
                                header += f" — {extra_preview}"

                        with st.expander(header, expanded=False):
                            if fields_used:
                                st.caption(f"Detected fields: {', '.join(fields_used)}")
                            if raw_values:
                                st.caption(f"Raw values captured: {', '.join(raw_values[:10])}{'…' if len(raw_values) > 10 else ''}")
                            # For each processed column in this row, show the three views + entities
                            for col_name, cdata in cols_data.items():
                                st.markdown(f"#### Field: `{col_name}`")
                                original_text = cdata.get('original', '')
                                anonymized_text = cdata.get('anonymized', '')
                                tokenized_text = cdata.get('tokenized', '')
                                ents = cdata.get('entities') or []

                                # Three-column layout similar to interactive view
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    self._render_styled_output("Original Text", original_text, key=f"batch_orig_{r_index}_{col_name}")
                                with c2:
                                    self._render_styled_output("Tokenized Text", tokenized_text, key=f"batch_tok_{r_index}_{col_name}")
                                with c3:
                                    self._render_styled_output("Anonymized Text", anonymized_text, key=f"batch_anon_{r_index}_{col_name}")

                                # Entities table and exposure details for this field
                                if ents:
                                    # Field exposure indicator using analyzer
                                    field_risk = cdata.get('exposure_level', 'NO')
                                    if field_risk == "FULL":
                                        st.error("Exposure risk: FULL — original PII appears in anonymized text")
                                    elif field_risk == "PARTIAL":
                                        st.warning("Exposure risk: PARTIAL — partial overlap with original PII detected")
                                    else:
                                        st.success("Exposure risk: NO — no original PII present in anonymized text")

                                    # Show per-entity exposure details if any
                                    exp_details = cdata.get('exposure_details') or []
                                    if exp_details:
                                        st.caption("Exposed entities (per-field):")
                                        st.dataframe(pd.DataFrame(exp_details), width="stretch")

                                    ent_rows = [
                                        {
                                            'entity_type': e.get('entity_type') or e.get('type'),
                                            'text': e.get('text'),
                                            'start': e.get('start'),
                                            'end': e.get('end'),
                                            'confidence': round(float(e.get('confidence', 0)), 3) if isinstance(e.get('confidence'), (int, float)) else e.get('confidence'),
                                            'recognizer': e.get('recognizer') or e.get('source') or '',
                                        }
                                        for e in ents
                                    ]
                                    st.dataframe(pd.DataFrame(ent_rows), width="stretch")
                                else:
                                    st.info("No entities detected in this field.")

                                # Non-PII retained content flagging
                                np_cnt = cdata.get('non_pii_retained_count', 0)
                                np_samples = cdata.get('non_pii_retained_samples', [])
                                if np_cnt > 0:
                                    st.info(f"Non-PII retained tokens present: {np_cnt} (e.g., {', '.join(np_samples)})")

                        shown += 1
                        if shown >= max_preview:
                            break

            # Optional: Full exposure analysis on this batch (in-memory)
            st.subheader("Full exposure analysis on this batch")
            use_preview_subset = st.checkbox(
                "Analyze only the previewed rows",
                value=False,
                help="If checked, runs on just the rows shown above. Otherwise runs on all processed rows in this batch."
            )
            if st.button("🔍 Run full exposure analysis on this batch", key="run_full_exp_batch"):
                try:
                    # Build minimal results structure expected by PIIExposureAnalyzer
                    indices = sorted(record_details.keys())
                    if use_preview_subset:
                        # Limit to the first max_preview rows we showed (respecting entities filter)
                        subset = []
                        shown = 0
                        only_with_entities = st.session_state.get("batch_detail_only_entities", True)
                        for r_index in indices:
                            rec = record_details[r_index]
                            total_entities = sum(len((cdata.get('entities') or [])) for cdata in rec.get('columns', {}).values())
                            if only_with_entities and total_entities == 0:
                                continue
                            subset.append(r_index)
                            shown += 1
                            if shown >= st.session_state.get("batch_detail_only_entities_max", 1000):
                                break
                        indices = subset

                    results_for_exposure: List[Dict[str, Any]] = []
                    for r_index in indices:
                        rec = record_details[r_index]
                        cols = rec.get('columns', {})
                        gt_fields: List[str] = []
                        gt_values: List[str] = []
                        anon_parts: List[str] = []
                        orig_parts: List[str] = []
                        for cname, cdata in cols.items():
                            ents = cdata.get('entities') or []
                            for e in ents:
                                txt = (e.get('text') or '').strip()
                                et = (e.get('entity_type') or e.get('type') or 'PII')
                                if txt:
                                    gt_fields.append(str(et))
                                    gt_values.append(txt)
                            anon_parts.append(cdata.get('anonymized', '') or '')
                            orig_parts.append(cdata.get('original', '') or '')
                        rec_obj = {
                            'record_id': r_index,
                            'ground_truth_pii': {
                                'fields_used': gt_fields,
                                'raw_values': gt_values,
                            },
                            'fields': {
                                'source': {
                                    'anonymized_text': ' '.join(anon_parts).strip(),
                                    'original_text': ' '.join(orig_parts).strip(),
                                }
                            }
                        }
                        results_for_exposure.append(rec_obj)

                    analyzer = PIIExposureAnalyzer()
                    summary = analyzer.analyze_all_records(results_for_exposure)
                    report = analyzer.generate_report(summary)

                    # Display results with existing viewer
                    self.display_exposure_results(summary)

                    # Downloads
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            label="Download exposure text report",
                            data=report,
                            file_name=f"exposure_report_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                        )
                    with c2:
                        st.download_button(
                            label="Download exposure JSON",
                            data=json.dumps(summary, indent=2),
                            file_name=f"exposure_analysis_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                        )
                except Exception as e:
                    st.error(f"Exposure analysis failed: {e}")

    @staticmethod
    def _safe_parse_list_field(value: Any) -> List[str]:
        """Parse list-like values stored as strings or collections into a list of strings."""
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value if v is not None and str(v).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text or text in {"[]", "()", "{}", "null", "None"}:
                return []
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parts = [part.strip().strip("\"'") for part in text.split(',')]
                return [part for part in parts if part]
            else:
                if isinstance(parsed, (list, tuple, set)):
                    return [str(v) for v in parsed if v is not None and str(v).strip()]
                return [str(parsed)] if parsed not in (None, '') else []
        return []

    def _normalize_results_for_exposure(self, raw_records: Any) -> List[Dict[str, Any]]:
        """Convert assorted result formats into the structure expected by PIIExposureAnalyzer."""
        if raw_records is None:
            return []

        if isinstance(raw_records, dict):
            if isinstance(raw_records.get('results'), list):
                records = [r for r in raw_records['results'] if isinstance(r, dict)]
            else:
                records = [raw_records]
        elif isinstance(raw_records, list):
            records = [r for r in raw_records if isinstance(r, dict)]
        else:
            return []

        if not records:
            return []

        first_record = next((r for r in records if isinstance(r, dict)), None)
        if first_record and isinstance(first_record.get('ground_truth_pii'), dict):
            return records

        grouped: Dict[str, Dict[str, Any]] = {}
        fallback_counter = 0

        def _merge_text(existing: str, new_text: str) -> str:
            """Append new_text to existing string if it's non-empty and not already included."""
            if not new_text:
                return existing
            if not existing:
                return new_text
            if new_text in existing:
                return existing
            return f"{existing} {new_text}".strip()

        for entry in records:
            raw_id = entry.get('record_id')
            if raw_id is None:
                raw_id = entry.get('row_index')
            if raw_id is None:
                raw_id = entry.get('rowId')
            if raw_id is None:
                raw_id = entry.get('row_number')
            if raw_id is None:
                raw_id = entry.get('id')
            if raw_id is None:
                raw_id = fallback_counter
                fallback_counter += 1

            record_key = str(raw_id)
            try:
                numeric_order = int(raw_id)
            except Exception:
                numeric_order = None

            group = grouped.setdefault(
                record_key,
                {
                    'record_id': record_key,
                    'ground_truth_pii': {'fields_used': [], 'raw_values': []},
                    'fields': {},
                    '_order': numeric_order,
                },
            )
            if group.get('_order') is None and numeric_order is not None:
                group['_order'] = numeric_order

            column_name = entry.get('column') or entry.get('field') or 'source'
            column_key = str(column_name)
            field_info = group['fields'].setdefault(
                column_key,
                {
                    'anonymized_text': '',
                    'original_text': '',
                    'tokenized_text': '',
                    'masked_text': '',
                },
            )

            tokenized_text = entry.get('tokenized_column_text') or entry.get('tokenized_text') or ''
            masked_text = entry.get('anonymized_column_text') or entry.get('anonymized_text') or ''
            anonymized_text = tokenized_text or masked_text
            original_text = entry.get('original_column_text') or entry.get('original_text') or ''

            field_info['tokenized_text'] = _merge_text(field_info['tokenized_text'], tokenized_text)
            field_info['masked_text'] = _merge_text(field_info['masked_text'], masked_text)
            field_info['anonymized_text'] = _merge_text(field_info['anonymized_text'], anonymized_text)
            field_info['original_text'] = _merge_text(field_info['original_text'], original_text)

            fields_used = self._safe_parse_list_field(entry.get('pii_fields_used'))
            raw_values = self._safe_parse_list_field(entry.get('pii_raw_values'))
            if fields_used and raw_values:
                gt = group['ground_truth_pii']
                if not gt['fields_used'] or len(fields_used) > len(gt['fields_used']):
                    n = min(len(fields_used), len(raw_values))
                    gt['fields_used'] = [str(v) for v in fields_used[:n]]
                    gt['raw_values'] = [str(v) for v in raw_values[:n]]

        sortable_records: List[tuple] = []

        for rec in grouped.values():
            fields = rec['fields']
            if 'source' not in fields or not isinstance(fields['source'], dict):
                combined_original = ' '.join(
                    v.get('original_text', '') for v in fields.values() if isinstance(v, dict) and v.get('original_text')
                ).strip()
                combined_anonymized = ' '.join(
                    v.get('anonymized_text', '') for v in fields.values() if isinstance(v, dict) and v.get('anonymized_text')
                ).strip()
                combined_tokenized = ' '.join(
                    v.get('tokenized_text', '') for v in fields.values() if isinstance(v, dict) and v.get('tokenized_text')
                ).strip()
                combined_masked = ' '.join(
                    v.get('masked_text', '') for v in fields.values() if isinstance(v, dict) and v.get('masked_text')
                ).strip()
                primary_anonymized = combined_tokenized or combined_anonymized or combined_masked
                fields['source'] = {
                    'original_text': combined_original,
                    'anonymized_text': primary_anonymized,
                }
                if combined_tokenized:
                    fields['source']['tokenized_text'] = combined_tokenized
                if combined_masked:
                    fields['source']['masked_text'] = combined_masked
            else:
                fields['source'].setdefault('original_text', '')
                fields['source'].setdefault('anonymized_text', '')
                if 'tokenized_text' in fields['source']:
                    fields['source']['anonymized_text'] = fields['source'].get('tokenized_text') or fields['source']['anonymized_text']
                fields['source'].setdefault('tokenized_text', '')
                fields['source'].setdefault('masked_text', '')

            rec['anonymized_text'] = fields['source']['anonymized_text']
            rec['original_text'] = fields['source']['original_text']
            if fields['source'].get('tokenized_text'):
                rec['tokenized_text'] = fields['source']['tokenized_text']
            if fields['source'].get('masked_text'):
                rec['masked_text'] = fields['source']['masked_text']

            gt = rec['ground_truth_pii']
            if gt['fields_used'] and gt['raw_values']:
                n = min(len(gt['fields_used']), len(gt['raw_values']))
                gt['fields_used'] = [str(v) for v in gt['fields_used'][:n]]
                gt['raw_values'] = [str(v) for v in gt['raw_values'][:n]]
            else:
                gt['fields_used'] = []
                gt['raw_values'] = []

            order_val = rec.pop('_order', None)
            sortable_records.append((order_val, rec['record_id'], rec))

        def _sort_key(item):
            order_val, record_id, _ = item
            if order_val is not None:
                return (0, order_val)
            try:
                return (1, int(record_id))
            except Exception:
                return (2, str(record_id))

        normalized = [rec for _, _, rec in sorted(sortable_records, key=_sort_key)]
        return normalized

    def _render_inline_results_and_exposure_section(self) -> None:
        """Inline viewer to open saved results files and run exposure analysis within Interactive UI."""
        st.subheader("📂 View saved results + Exposure analysis")
        st.caption("Browse saved JSON or CSV outputs from results/; run exposure analysis on JSON files when available.")

        json_files = self.get_json_results_files()
        csv_files = self.get_csv_results_files()
        if not json_files and not csv_files:
            st.info("No results found in results/. Run an analysis or interactive batch to create one.")
            return

        tab_json, tab_csv = st.tabs(["JSON results", "CSV results"])

        with tab_json:
            if not json_files:
                st.info("No JSON results found. Enable saving to results/ during batch analysis to generate one.")
            else:
                names = [p.name for p in json_files]
                default_index = 0
                last_name = st.session_state.get('__last_results_filename__')
                if last_name in names:
                    try:
                        default_index = names.index(last_name)
                    except Exception:
                        default_index = 0
                sel_name = st.selectbox("Results JSON file", options=names, index=default_index, key="results_json_select")
                if sel_name:
                    st.session_state['__last_results_filename__'] = sel_name
                    sel_path = self.results_path / sel_name
                    results = self.load_json_results(sel_path)
                    if not results:
                        st.warning("Unable to load the selected JSON file.")
                    else:
                        with st.expander("Quick summary", expanded=True):
                            try:
                                self.render_results_summary(results)
                            except Exception:
                                st.write("Summary unavailable for this file structure.")

                        with st.expander("Entity overview", expanded=False):
                            try:
                                self.render_results_overview(results)
                            except Exception:
                                st.write("Overview unavailable for this file structure.")

                        with st.expander("🧾 Raw JSON data", expanded=False):
                            st.json(results)

                        run_exp = st.button("🔍 Run Exposure Analysis on this file", key="run_exposure_inline")
                        if run_exp:
                            try:
                                analyzer = PIIExposureAnalyzer()
                                data_for_analysis = results.get('results', results)
                                normalized_records = self._normalize_results_for_exposure(data_for_analysis)
                                if not normalized_records:
                                    st.warning("This results file is missing the expected source, pii_fields_used, and pii_raw_values data needed for exposure analysis.")
                                else:
                                    summary = analyzer.analyze_all_records(normalized_records)
                                    st.session_state.exposure_analysis_results = summary
                                    self.display_exposure_results(summary)
                                    report = analyzer.generate_report(summary)
                                    st.session_state.exposure_report = report
                                    st.download_button(
                                        label="Download exposure text report",
                                        data=report,
                                        file_name=f"exposure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                        mime="text/plain",
                                        key="download_exposure_txt"
                                    )
                                    st.download_button(
                                        label="Download exposure JSON",
                                        data=json.dumps(summary, indent=2),
                                        file_name=f"exposure_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json",
                                        key="download_exposure_json"
                                    )
                            except Exception as e:
                                st.error(f"Exposure analysis failed: {e}")

        with tab_csv:
            if not csv_files:
                st.info("No CSV results found. Enable saving to results/ during batch analysis to generate one.")
            else:
                csv_names = [p.name for p in csv_files]
                default_csv_index = 0
                last_csv = st.session_state.get('__last_results_csv__')
                if last_csv in csv_names:
                    try:
                        default_csv_index = csv_names.index(last_csv)
                    except Exception:
                        default_csv_index = 0
                sel_csv_name = st.selectbox("Results CSV file", options=csv_names, index=default_csv_index, key="results_csv_select")
                if sel_csv_name:
                    st.session_state['__last_results_csv__'] = sel_csv_name
                    sel_csv_path = self.results_path / sel_csv_name
                    try:
                        df_csv_view = pd.read_csv(sel_csv_path)
                        preview_rows = min(len(df_csv_view), 500)
                        st.dataframe(df_csv_view.head(preview_rows), width="stretch")
                        st.download_button(
                            label="Download selected CSV",
                            data=sel_csv_path.read_bytes(),
                            file_name=sel_csv_name,
                            mime="text/csv",
                            key="download_saved_csv"
                        )
                    except Exception as e:
                        st.error(f"Could not load CSV: {e}")
    
    def _render_interactive_output(self, payload: Dict[str, Any]) -> None:
        """Render interactive search results persisted in session state."""
        if not payload:
            return

        text: str = payload.get('text', '') or ''
        results: Dict[str, Any] = payload.get('results') or {}
        config: Dict[str, Any] = payload.get('config') or {}
        debug_config: Dict[str, Any] = payload.get('debug_config') or {}
        warnings: List[str] = payload.get('warnings') or []
        used_rows: List[Dict[str, Any]] = payload.get('used_recognizers') or []
        anonymize_results: bool = payload.get('anonymize_results', True)
        max_results: int = int(payload.get('max_results', 20))
        selected_entities: List[str] = payload.get('selected_entities') or []
        spacy_on: bool = bool(payload.get('spacy_on'))
        detected_spans: List[Dict[str, Any]] = payload.get('detected_spans') or self._collect_token_spans(text, results)
        anonymized_text: str = results.get('anonymized_text', text)
        tokenized_text: str = results.get('tokenized_text', text)

        if debug_config:
            with st.expander("🔧 Debug Configuration", expanded=False):
                st.json({k: v for k, v in debug_config.items() if v is not None})

        badge = "<span style='background:#2e7d32;color:#fff;padding:3px 8px;border-radius:12px;font-size:12px;font-weight:600;'>spaCy NER: ON</span>" if spacy_on else "<span style='background:#b71c1c;color:#fff;padding:3px 8px;border-radius:12px;font-size:12px;font-weight:600;'>spaCy NER: OFF</span>"
        st.markdown(badge, unsafe_allow_html=True)

        for msg in warnings:
            st.warning(msg)

        if used_rows:
            with st.expander("🧩 Used recognizers (current analyzer registry)", expanded=False):
                for row in used_rows:
                    st.markdown(f"- {row['class']} • {row['source']} • entities: {row['entities']}")

        try:
            if config.get('__use_yaml_patterns__'):
                subset = payload.get('yaml_entity_subset') or config.get('__yaml_entity_subset__') or []
                yfile = payload.get('yaml_preview_path') or config.get('__yaml_recognizers_path__')
                with st.expander("📄 YAML recognizers loaded", expanded=False):
                    if yfile:
                        try:
                            y_path = Path(yfile)
                            if not y_path.is_absolute():
                                y_path = (self.base_path / y_path).resolve()
                        except Exception:
                            y_path = self.recognizers_config_path
                    else:
                        y_path = self.recognizers_config_path
                    if y_path.exists():
                        self._render_yaml_entries_preview(y_path, subset if isinstance(subset, list) else [])
        except Exception:
            pass

        if anonymize_results:
            st.subheader("🔒 Output")
            c_orig, c_an, c_tok = st.columns(3)
            with c_orig:
                self._render_styled_output("Original Text", text, key="interactive_original")
            with c_an:
                self._render_styled_output("Anonymized Text", anonymized_text, key="interactive_anonymized")
            with c_tok:
                self._render_styled_output("Tokenized Text", tokenized_text, key="interactive_tokenized")

            st.subheader("🖍️ PII highlighting")
            if 'hl_pii_in_original' not in st.session_state:
                st.session_state['hl_pii_in_original'] = True
            hl_enabled = st.checkbox("Highlight PII in original text", key="hl_pii_in_original")

            spans_for_use = detected_spans

            if hl_enabled and spans_for_use:
                self._render_highlighted_text("Original (highlight PII)", text, spans_for_use)
                etypes = []
                seen_types = set()
                for s in spans_for_use:
                    t = (s.get('entity_type') or 'PII').upper()
                    if t not in seen_types:
                        seen_types.add(t)
                        etypes.append(t)
                if etypes:
                    chips = []
                    for t in etypes[:12]:
                        col = self._color_for_entity_type(t)
                        chips.append(f"<span style='display:inline-block;background:{col};color:#fff;border-radius:6px;padding:2px 6px;margin:2px 6px 0 0;font-size:11px;'>{self._escape_html(t)}</span>")
                    st.markdown("<div style='margin-top:4px'>" + ''.join(chips) + "</div>", unsafe_allow_html=True)

                if st.checkbox("Preview anonymization/tokenization using selected spans", value=False, key="preview_selected_spans"):
                    try:
                        preview_tok = self._tokenize_from_detections(text, spans_for_use)
                    except Exception:
                        preview_tok = text
                    preview_anon = None
                    if config.get('use_presidio'):
                        try:
                            from pii_analysis_cli import load_presidio_engines
                            _, preview_anonymizer = load_presidio_engines(config)
                        except Exception:
                            preview_anonymizer = None
                    else:
                        preview_anonymizer = None

                    if preview_anonymizer is not None:
                        try:
                            from presidio_analyzer import RecognizerResult
                            rr = []
                            for d in spans_for_use:
                                try:
                                    rr.append(RecognizerResult(entity_type=d['entity_type'], start=int(d['start']), end=int(d['end']), score=float(d.get('confidence') or 0.5)))
                                except Exception:
                                    pass
                            if rr:
                                unique, seen = [], set()
                                for r in rr:
                                    key = (r.start, r.end, r.entity_type)
                                    if key not in seen:
                                        unique.append(r)
                                        seen.add(key)
                                preview_anon = preview_anonymizer.anonymize(text=text, analyzer_results=unique).text
                        except Exception:
                            preview_anon = None
                    if preview_anon is None:
                        preview_anon = self._fallback_anonymize_from_detections(text, spans_for_use)

                    c1, c2 = st.columns(2)
                    with c1:
                        self._render_styled_output("Anonymized (preview from selected spans)", preview_anon, key="preview_anonymized")
                    with c2:
                        self._render_styled_output("Tokenized (preview from selected spans)", preview_tok, key="preview_tokenized")

            try:
                exp = PIIExposureAnalyzer()
                ents = []
                for d in spans_for_use:
                    try:
                        ents.append({'text': text[int(d['start']):int(d['end'])], 'entity_type': d.get('entity_type'), 'start': d.get('start'), 'end': d.get('end')})
                    except Exception:
                        pass

                per_entity_rows = []
                worst = 'NO'

                def _rank(v: str) -> int:
                    return {'NO': 0, 'PARTIAL': 1, 'FULL': 2}.get(v, 0)

                for e in ents:
                    etxt = (e.get('text') or '').strip()
                    if not etxt:
                        continue
                    level, parts = exp.check_exposure_level(etxt, anonymized_text)
                    lvl = 'NO' if level == 'NO_EXPOSURE' else ('FULL' if level == 'FULL_EXPOSURE' else 'PARTIAL')
                    worst = lvl if _rank(lvl) > _rank(worst) else worst
                    if lvl != 'NO':
                        per_entity_rows.append({
                            'entity_type': e.get('entity_type'),
                            'text': etxt,
                            'level': lvl,
                            'exposed_snippets': parts
                        })

                mask = [True] * len(text)
                for e in ents:
                    s, t = e.get('start'), e.get('end')
                    if isinstance(s, int) and isinstance(t, int) and 0 <= s < t <= len(text):
                        for i in range(s, t):
                            mask[i] = False
                non_pii_text = ''.join(ch if mask[i] else ' ' for i, ch in enumerate(text))
                orig_tokens = exp.tokenize_text(non_pii_text)
                anon_tokens = exp.tokenize_text(anonymized_text)
                COMMON_WORDS = {'the','and','for','with','this','that','from','have','has','are','was','were','you','your','his','her','their','our','not','but','all','any','can','will','would','there','here','about','into','over','under','more','most','some','such','only','other','than','then','also','between','after','before','because','where','when','which','while','on','in','at','to','of','a','an','is','it','as','by','be'}

                def _keep(tok: str) -> bool:
                    return ((tok.isdigit() and len(tok) >= 4) or (len(tok) >= 3 and tok.isalpha())) and tok not in COMMON_WORDS

                retained = sorted(t for t in (orig_tokens & anon_tokens) if _keep(t))

                st.subheader("🛡️ Exposure check (this record)")
                if worst == 'FULL':
                    st.error("Exposure risk: FULL — original PII appears in anonymized text")
                elif worst == 'PARTIAL':
                    st.warning("Exposure risk: PARTIAL — partial overlap with original PII detected")
                else:
                    st.success("Exposure risk: NO — no original PII present in anonymized text")

                if per_entity_rows:
                    st.caption("Exposed entities:")
                    st.dataframe(pd.DataFrame(per_entity_rows), width="stretch")

                if retained:
                    if st.checkbox("Highlight retained non-PII tokens in anonymized text", value=False, key="hl_nonpii_interactive"):
                        try:
                            hi = anonymized_text
                            for tok in sorted(retained, key=len, reverse=True)[:50]:
                                hi = re.sub(rf"\\b{re.escape(tok)}\\b", f"[[{tok}]]", hi, flags=re.IGNORECASE)
                            self._render_styled_output("Anonymized (highlight non-PII)", hi, key="interactive_anonymized_hl")
                        except Exception:
                            pass
                    else:
                        st.info(f"Non-PII retained tokens present: {len(retained)} (e.g., {', '.join(retained[:10])})")
            except Exception:
                pass

        self._display_search_results(text, results, anonymize_results, max_results)

    def _perform_interactive_search(self, text: str, use_transformers: bool, use_ollama: bool, sensitivity: float, selected_entities: List[str], anonymize_results: bool, max_results: int, use_custom_datetime: bool, transformer_models: Optional[List[str]] = None):
        """Perform interactive PII search and persist results for reruns."""
        try:
            from pii_analysis_cli import load_presidio_engines, enhanced_pii_analysis

            warnings: List[str] = []

            with st.spinner("🔍 Searching for PII entities..."):
                config: Dict[str, Any] = {}
                if self.config_path.exists():
                    with open(self.config_path, 'r') as f:
                        config = yaml.safe_load(f) or {}

                config['sensitivity'] = sensitivity
                config['use_transformers'] = use_transformers
                config['use_ollama'] = use_ollama
                config['use_custom_datetime'] = use_custom_datetime
                if use_transformers and transformer_models:
                    config['transformer_models'] = transformer_models

                disable_presidio = st.session_state.get('__disable_presidio__')
                if disable_presidio is None:
                    disable_presidio = False
                config['use_presidio'] = not disable_presidio

                disable_builtin = st.session_state.get('__disable_builtin_entities__', [])
                enable_only_builtin = st.session_state.get('__enable_only_builtin_entities__', [])
                if disable_builtin:
                    config['__disable_builtin_entities__'] = disable_builtin
                if enable_only_builtin:
                    config['__enable_only_builtin_entities__'] = enable_only_builtin

                if st.session_state.get('__built_in_only__'):
                    config['use_uk_patterns'] = False
                    config['__use_yaml_patterns__'] = False
                    config['__use_code_recognizers__'] = False
                    config['__built_in_only__'] = True
                else:
                    config['use_uk_patterns'] = True
                    config['__use_yaml_patterns__'] = st.session_state.get('__use_yaml_patterns__', True)
                    config['__use_code_recognizers__'] = st.session_state.get('__use_code_recognizers__', True)
                    config['__yaml_entity_subset__'] = st.session_state.get('__yaml_entity_subset__', [])
                    config['__code_entity_subset__'] = st.session_state.get('__code_entity_subset__', [])
                    yaml_path = st.session_state.get('__yaml_recognizers_path__')
                    if yaml_path:
                        config['__yaml_recognizers_path__'] = yaml_path
                config['__interactive_mode__'] = True

                debug_config = {
                    'use_presidio': config.get('use_presidio'),
                    'use_transformers': config.get('use_transformers'),
                    'transformer_models': config.get('transformer_models'),
                    'use_ollama': config.get('use_ollama'),
                    'use_uk_patterns': config.get('use_uk_patterns'),
                    'yaml_patterns': config.get('__use_yaml_patterns__'),
                    'yaml_file': config.get('__yaml_recognizers_path__'),
                    'code_recognizers': config.get('__use_code_recognizers__'),
                    'disable_builtin_entities': config.get('__disable_builtin_entities__'),
                    'enable_only_builtin_entities': config.get('__enable_only_builtin_entities__'),
                    'interactive_mode': config.get('__interactive_mode__')
                }

                init_error_trace = None
                try:
                    analyzer, anonymizer = load_presidio_engines(config)
                except Exception as init_err:
                    import traceback
                    init_error_trace = traceback.format_exc()
                    st.error(f"Engine initialization failed: {init_err}")
                    analyzer, anonymizer = None, None

                if config.get('use_presidio') and not analyzer:
                    st.error("❌ Failed to initialize Presidio analyzer")
                    if init_error_trace:
                        with st.expander("Show engine initialization traceback"):
                            st.code(init_error_trace)
                    return

                if use_transformers:
                    try:
                        from transformer_integration import get_analyzer_with_transformers
                        built_in_only_mode = st.session_state.get('__built_in_only__', False)
                        add_custom = None
                        if not built_in_only_mode:
                            try:
                                from uk_recognizers import register_uk_recognizers  # type: ignore
                                add_custom = register_uk_recognizers
                            except Exception:
                                pass
                        tm = transformer_models if transformer_models else ["dslim/bert-base-NER"]
                        tm = list(dict.fromkeys(tm))
                        analyzer = get_analyzer_with_transformers(
                            sensitivity=sensitivity,
                            uk_specific=True,
                            transformer_model=tm[0],
                            transformer_models=tm,
                            add_custom_recognizers=add_custom,
                            use_custom_datetime=use_custom_datetime
                        )
                    except Exception as e:
                        warnings.append(f"⚠️ Transformer analyzer could not be initialized: {e}")

                spacy_on = False
                try:
                    if analyzer is not None and getattr(analyzer, 'registry', None):
                        for r in getattr(analyzer.registry, 'recognizers', []) or []:
                            if type(r).__name__ == 'SpacyRecognizer':
                                spacy_on = True
                                break
                except Exception:
                    spacy_on = False

                results = enhanced_pii_analysis(
                    text=text,
                    analyzer=analyzer,
                    config=config,
                    selected_entities=selected_entities if selected_entities else None
                )

                used_rows: List[Dict[str, Any]] = []
                try:
                    used_rows = self._summarize_registry(analyzer)
                except Exception:
                    used_rows = []

                detected_spans = self._collect_token_spans(text, results)

                tokenized_text = text
                anonymized_text = text

                if anonymize_results:
                    try:
                        tokenized_text = self._tokenize_from_detections(text, detected_spans)
                    except Exception as e:
                        warnings.append(f"⚠️ Tokenization failed: {e}; falling back to masking.")
                        tokenized_text = self._fallback_anonymize(text, results)

                    anonymized_text = None
                    if config.get('use_presidio') and anonymizer:
                        try:
                            from presidio_analyzer import RecognizerResult
                            rr = []
                            for d in detected_spans:
                                try:
                                    rr.append(RecognizerResult(entity_type=d['entity_type'], start=int(d['start']), end=int(d['end']), score=float(d.get('confidence') or 0.5)))
                                except Exception:
                                    pass
                            if rr:
                                unique, seen = [], set()
                                for r in rr:
                                    key = (r.start, r.end, r.entity_type)
                                    if key not in seen:
                                        unique.append(r)
                                        seen.add(key)
                                anonymized_text = anonymizer.anonymize(text=text, analyzer_results=unique).text
                            elif analyzer:
                                analysis_results = analyzer.analyze(
                                    text=text,
                                    language='en',
                                    entities=selected_entities if selected_entities else None,
                                    score_threshold=sensitivity if sensitivity else None,
                                )
                                anonymized_text = anonymizer.anonymize(text=text, analyzer_results=analysis_results).text
                        except Exception as e:
                            warnings.append(f"⚠️ Presidio anonymization failed: {e}; using fallback masking.")
                    if anonymized_text is None:
                        anonymized_text = self._fallback_anonymize(text, results)

                    results['anonymized_text'] = anonymized_text
                    results['tokenized_text'] = tokenized_text

                payload: Dict[str, Any] = {
                    'text': text,
                    'config': json.loads(json.dumps(config, default=str)) if config else {},
                    'results': results,
                    'used_recognizers': used_rows,
                    'yaml_preview_path': config.get('__yaml_recognizers_path__'),
                    'yaml_entity_subset': config.get('__yaml_entity_subset__', []),
                    'anonymize_results': anonymize_results,
                    'max_results': max_results,
                    'selected_entities': selected_entities,
                    'warnings': warnings,
                    'spacy_on': spacy_on,
                    'detected_spans': detected_spans,
                    'debug_config': debug_config,
                    'timestamp': time.time(),
                }

                st.session_state['__interactive_last_payload__'] = payload

        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")
            if st.checkbox("Show error details", key="show_error_details"):
                st.code(str(e))
    
    def _perform_enhanced_analysis(self, text: str, use_transformers: bool, use_ollama: bool, sensitivity: float, selected_entities: List[str], anonymize_results: bool, max_results: int, use_custom_datetime: bool, transformer_models: Optional[List[str]] = None):
        """Perform enhanced analysis showing individual recognizer results"""
        try:
            from enhanced_recognizer_analysis import EnhancedRecognizerAnalysis, display_enhanced_analysis_results
            
            with st.spinner("🔬 Running enhanced recognizer analysis..."):
                # Load configuration
                config = {}
                if self.config_path.exists():
                    with open(self.config_path, 'r') as f:
                        config = yaml.safe_load(f)
                
                # Update config with current settings
                config['sensitivity'] = sensitivity
                config['use_transformers'] = use_transformers
                config['use_ollama'] = use_ollama
                config['use_custom_datetime'] = use_custom_datetime
                # Respect disable-presidio checkbox (default True if not disabled)
                disable_presidio = st.session_state.get('__disable_presidio__', False)
                config['use_presidio'] = not disable_presidio
                built_in_only = st.session_state.get('__built_in_only__', False)
                if built_in_only:
                    # Explicitly disable non-built-in recognizers
                    config['use_uk_patterns'] = False
                    config['__use_yaml_patterns__'] = False
                    config['__use_code_recognizers__'] = False
                else:
                    config['use_uk_patterns'] = True
                    config['__use_yaml_patterns__'] = st.session_state.get('__use_yaml_patterns__', True)
                    config['__use_code_recognizers__'] = st.session_state.get('__use_code_recognizers__', True)
                if use_transformers and transformer_models:
                    config['transformer_models'] = transformer_models
                
                with st.expander("🔧 Enhanced Analysis Configuration", expanded=False):
                    st.json({k: v for k, v in {
                        'use_presidio': config.get('use_presidio'),
                        'use_transformers': config.get('use_transformers'),
                        'transformer_models': config.get('transformer_models'),
                        'use_ollama': config.get('use_ollama'),
                        'use_uk_patterns': config.get('use_uk_patterns'),
                        'built_in_only': built_in_only,
                        'yaml_patterns_enabled': config.get('__use_yaml_patterns__'),
                        'code_recognizers_enabled': config.get('__use_code_recognizers__'),
                        'sensitivity': config.get('sensitivity')
                    }.items() if v is not None})
                
                # Run enhanced analysis
                analyzer = EnhancedRecognizerAnalysis()
                results = analyzer.analyze_with_all_recognizers(text, config)
                
                # Display comprehensive results
                display_enhanced_analysis_results(results)
                
                # Do not render anonymized/tokenized blocks here to avoid duplicate green boxes; Interactive Search shows them once.
                
        except Exception as e:
            st.error(f"❌ Enhanced analysis failed: {str(e)}")
            if st.checkbox("Show enhanced analysis error details", key="show_enhanced_error_details"):
                st.code(str(e))
    
    def _create_anonymized_output(self, text: str, detections: List, config: Dict) -> None:
        """Create anonymized text from detections honoring Presidio disable flag"""
        try:
            # If Presidio disabled, use fallback masking directly
            if not config.get('use_presidio'):
                masked = self._fallback_anonymize_from_detections(text, detections)
                self._render_styled_output("Anonymized Text (fallback)", masked, key="enhanced_anon_fallback")
                return

            from pii_analysis_cli import load_presidio_engines
            from presidio_analyzer import RecognizerResult

            # Load anonymizer with current config so early short-circuit is respected
            _, anonymizer = load_presidio_engines(config)

            if anonymizer and detections:
                recognizer_results = [
                    RecognizerResult(
                        entity_type=d.entity_type,
                        start=d.start,
                        end=d.end,
                        score=float(d.confidence)
                    ) for d in detections
                ]

                # Deduplicate by span
                unique_results = []
                seen_positions = set()
                for r in recognizer_results:
                    key = f"{r.start}-{r.end}"
                    if key not in seen_positions:
                        unique_results.append(r)
                        seen_positions.add(key)

                anonymized_result = anonymizer.anonymize(text=text, analyzer_results=unique_results)

                self._render_styled_output("Anonymized Text", anonymized_result.text, key="enhanced_anon_std")
            else:
                # Fallback if anonymizer missing
                masked = self._fallback_anonymize_from_detections(text, detections)
                self._render_styled_output("Anonymized Text (fallback)", masked, key="enhanced_anon_fallback2")
        except Exception as e:
            st.warning(f"⚠️ Anonymization failed: {str(e)}")

    def _get_anonymized_text(self, text: str, detections: List, config: Dict) -> str:
        """Return anonymized text string using Presidio if enabled, else fallback masking."""
        try:
            if not config.get('use_presidio'):
                return self._fallback_anonymize_from_detections(text, detections)
            from pii_analysis_cli import load_presidio_engines
            from presidio_analyzer import RecognizerResult
            _, anonymizer = load_presidio_engines(config)
            if anonymizer and detections:
                rr = []
                for d in detections:
                    try:
                        rr.append(RecognizerResult(entity_type=d.entity_type, start=int(d.start), end=int(d.end), score=float(d.confidence)))
                    except AttributeError:
                        rr.append(RecognizerResult(entity_type=d.get('entity_type'), start=int(d.get('start')), end=int(d.get('end')), score=float(d.get('confidence') or 0.5)))
                # Dedup by span and entity type
                unique, seen = [], set()
                for r in rr:
                    k = (r.start, r.end, r.entity_type)
                    if k not in seen:
                        unique.append(r)
                        seen.add(k)
                return anonymizer.anonymize(text=text, analyzer_results=unique).text
        except Exception:
            pass
        return self._fallback_anonymize_from_detections(text, detections)

    def _render_styled_output(self, title: str, value: str, key: str) -> None:
        """Render colour-coded text blocks with high-contrast typography."""
        safe = self._escape_html(value if value is not None else "")
        title_lower = (title or "").lower()

        palette = {
            'original': {
                'bg': '#111827',
                'fg': '#F9FAFB',
                'border': '#1F2937',
                'accent': '#60A5FA',
            },
            'tokenized': {
                'bg': '#312E81',
                'fg': '#E0E7FF',
                'border': '#4338CA',
                'accent': '#A855F7',
            },
            'anonymized': {
                'bg': '#064E3B',
                'fg': '#ECFDF5',
                'border': '#047857',
                'accent': '#34D399',
            },
            'default': {
                'bg': '#1F2933',
                'fg': '#F5F7FA',
                'border': '#27323F',
                'accent': '#9CA3AF',
            },
        }

        if 'original' in title_lower:
            theme = palette['original']
        elif 'tokenized' in title_lower:
            theme = palette['tokenized']
        elif 'anonymized' in title_lower or 'anonymised' in title_lower:
            theme = palette['anonymized']
        else:
            theme = palette['default']

        st.markdown(
            dedent(
                f"""
                <div style="margin-bottom:14px;">
                    <div style="font-size:12px; letter-spacing:0.08em; text-transform:uppercase; font-weight:600; color:{theme['accent']}; margin-bottom:6px;">
                        {self._escape_html(title)}
                    </div>
                    <div style="background:{theme['bg']}; color:{theme['fg']}; border:1px solid {theme['border']}; border-radius:12px; padding:14px; white-space:pre-wrap; font-family:ui-monospace,Menlo,Monaco,Consolas,\"Liberation Mono\",\"Courier New\",monospace; font-size:13px; line-height:1.55;">
                        {safe}
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    def _escape_html(self, s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _merge_spans(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping/adjacent spans; expects dicts with start/end/entity_type (optional)."""
        norm: List[Dict[str, Any]] = []
        for s in spans or []:
            try:
                start = int(s.get('start')) if isinstance(s, dict) else int(getattr(s, 'start'))
                end = int(s.get('end')) if isinstance(s, dict) else int(getattr(s, 'end'))
            except Exception:
                continue
            if start is None or end is None or start >= end:
                continue
            et = s.get('entity_type') if isinstance(s, dict) else getattr(s, 'entity_type', None)
            norm.append({'start': start, 'end': end, 'entity_type': et})
        norm.sort(key=lambda x: (x['start'], x['end']))
        merged: List[Dict[str, Any]] = []
        for s in norm:
            if not merged or s['start'] > merged[-1]['end']:
                merged.append(dict(s))
            else:
                merged[-1]['end'] = max(merged[-1]['end'], s['end'])
                if merged[-1].get('entity_type') in (None, 'PII') and s.get('entity_type'):
                    merged[-1]['entity_type'] = s.get('entity_type')
        return merged

    def _color_for_entity_type(self, entity_type: Optional[str]) -> str:
        """Return a consistent highlight color for the given entity type."""
        palette = {
            'CREDIT_CARD': '#e53935',
            'DATE_TIME': '#fb8c00',
            'EMAIL_ADDRESS': '#8e24aa',
            'PHONE_NUMBER': '#3949ab',
            'PERSON': '#1e88e5',
            'NAME': '#1e88e5',
            'UK_POSTCODE': '#00897b',
            'UK_NATIONAL_INSURANCE': '#6d4c41',
            'UK_SORT_CODE': '#5e35b1',
            'UK_PHONE': '#039be5',
            'UK_BANK_ACCOUNT': '#5d4037',
            'UK_VAT_NUMBER': '#6a1b9a',
            'UK_COMPANY_NUMBER': '#1b5e20',
            'UK_PASSPORT': '#283593',
            'UK_VEHICLE_REG': '#546e7a',
            'IBAN_CODE': '#00796b',
            'CUSTOMER_ID': '#8d6e63',
            'US_SSN': '#b71c1c',
            'US_PASSPORT': '#0d47a1',
        }
        return palette.get((entity_type or '').upper(), '#546e7a')

    def _render_highlighted_text(self, title: str, text: str, spans: List[Dict[str, Any]]) -> None:
        """Render original text with PII spans highlighted using inline HTML styles."""
        spans = self._merge_spans(spans)
        parts: List[str] = []
        last = 0
        for s in spans:
            stt, end = s['start'], s['end']
            if last < stt:
                parts.append(self._escape_html(text[last:stt]))
            seg = text[stt:end]
            col = self._color_for_entity_type(s.get('entity_type'))
            parts.append(f"<span style='background:{col};color:#fff;border-radius:3px;padding:0 2px'>{self._escape_html(seg)}</span>")
            last = end
        if last < len(text):
            parts.append(self._escape_html(text[last:]))
        st.markdown(f"<div style='margin-bottom:6px;font-weight:600;'>{self._escape_html(title)}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:#f5f5f5;color:#000;border-radius:8px;padding:10px;white-space:pre-wrap;font-family:ui-monospace,Menlo,Monaco,Consolas,\"Liberation Mono\",\"Courier New\",monospace;font-size:13px;'>{''.join(parts)}</div>",
            unsafe_allow_html=True,
        )

    def _is_likely_pan(self, segment: str) -> bool:
        """Heuristic payment card detector using length and Luhn checksum."""
        if not segment:
            return False
        digits = ''.join(ch for ch in segment if ch.isdigit())
        if not (13 <= len(digits) <= 19):
            return False
            import re as _re
            total = 0
            alt = False
            # Avoid obvious UK phone shapes
            if _re.search(r"\b(?:\+?44\s?\d{9,11}|0\d{3,4}[\s-]?\d{3}[\s-]?\d{3,4})\b", segment):
                return False
            # Card-like grouping (#### #### #### #### or with hyphens)
            if _re.search(r"(?:\d{4}[ -]?){3}\d{4}", segment):
                return True
            for ch in reversed(digits):
                d = ord(ch) - 48
                if alt:
                    d *= 2
                    if d > 9:
                        d -= 9
                total += d
                alt = not alt
            if (total % 10) == 0:
                return True
            # As a fallback, accept contiguous 15/16 digit sequences as plausible PANs
            return len(digits) in (15, 16)

    def _collect_token_spans(self, text: str, results: Dict) -> List[Dict[str, Any]]:
        """Collect span dicts (start, end, entity_type) from all finding sources.
        Respects built-in-only mode by excluding UK pattern recognizers when enabled.
        """
        built_in_only = st.session_state.get('__built_in_only__', False)
        spans: List[Dict[str, Any]] = []
        sources = [
            ('presidio_findings', True),
            ('transformer_findings', True),
            ('ollama_findings', True),
            ('uk_pattern_findings', not built_in_only),
        ]
        for key, allowed in sources:
            if not allowed:
                continue
            for f in results.get(key, []) or []:
                try:
                    start = int(f.get('start') if isinstance(f, dict) else getattr(f, 'start', None))
                    end = int(f.get('end') if isinstance(f, dict) else getattr(f, 'end', None))
                    et = (f.get('entity_type') if isinstance(f, dict) else getattr(f, 'entity_type', None)) or (f.get('type') if isinstance(f, dict) else None)
                    if start is None or end is None or start >= end:
                        continue
                    if start < 0 or end > len(text):
                        continue
                    ent_txt = text[start:end]
                    if self._is_likely_pan(ent_txt):
                        et = 'CREDIT_CARD'
                    spans.append({'start': start, 'end': end, 'entity_type': str(et or 'PII')})
                except Exception:
                    continue
        # Deduplicate by (start,end)
        unique = []
        seen = set()
        for s in sorted(spans, key=lambda x: (x['start'], x['end'])):
            k = (s['start'], s['end'])
            if k not in seen:
                unique.append(s)
                seen.add(k)
        return unique

    def _fallback_anonymize_from_detections(self, text: str, detections: List) -> str:
        """Simple masking when Presidio is disabled using detection spans."""
        if not detections:
            # No detections: tokenize entire text so output is clearly tokenized
            import random, string
            def _full_tokenize(s: str) -> str:
                out = []
                for ch in s:
                    if ch.isalpha():
                        out.append(random.choice(string.ascii_uppercase if ch.isupper() else string.ascii_lowercase))
                    elif ch.isdigit():
                        out.append(random.choice(string.digits))
                    else:
                        out.append(ch)
                return ''.join(out)
            return _full_tokenize(text)
        # Build span list
        spans = []
        for d in detections:
            try:
                spans.append((d.start, d.end, d.entity_type))
            except AttributeError:
                # dict-like fallback
                spans.append((d.get('start'), d.get('end'), d.get('entity_type')))
        # Sort and collapse overlaps (prefer earliest, then longest)
        spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
        merged = []
        for s in spans:
            if not merged or s[0] >= merged[-1][1]:
                merged.append(list(s))
            else:
                # Overlap: extend end if needed
                if s[1] > merged[-1][1]:
                    merged[-1][1] = s[1]
        # Reconstruct
        masked_parts = []
        last = 0
        for start, end, etype in merged:
            if start is None or end is None:
                continue
            if last < start:
                masked_parts.append(text[last:start])
            token = f"<{etype.upper()}>" if etype else "<PII>"
            masked_parts.append(token)
            last = end
        if last < len(text):
            masked_parts.append(text[last:])
        return ''.join(masked_parts)

    def _tokenize_from_detections(self, text: str, detections: List) -> str:
        """Format-preserving tokenization for detected spans.
        - Letters -> random letters, preserving original per-char case
    - Digits -> replaced with random digits; punctuation/whitespace preserved
    - PAN/CREDIT_CARD: keep first 6 and last 4 digits, middle digits -> random uppercase letters
        """
        if not detections:
            return text
        import random, string

        # Prepare spans sorted by start
        norm_dets = []
        for d in detections:
            try:
                norm_dets.append((int(d.start), int(d.end), str(d.entity_type)))
            except AttributeError:
                norm_dets.append((int(d.get('start')), int(d.get('end')), str(d.get('entity_type'))))
        norm_dets = [s for s in norm_dets if s[0] is not None and s[1] is not None and s[0] < s[1]]
        norm_dets.sort(key=lambda x: (x[0], x[1]))

        # Merge overlaps to avoid double processing
        merged: List[List] = []
        for s in norm_dets:
            if not merged or s[0] >= merged[-1][1]:
                merged.append([s[0], s[1], s[2]])
            else:
                # extend end and keep broader entity type (prefer specific like CREDIT_CARD)
                merged[-1][1] = max(merged[-1][1], s[1])
                if 'CREDIT_CARD' in s[2] and 'CREDIT_CARD' not in merged[-1][2]:
                    merged[-1][2] = s[2]
        # If nothing to tokenize after merging, tokenize entire text
        if not merged:
            import random, string
            def _full_tokenize(s: str) -> str:
                out = []
                for ch in s:
                    if ch.isalpha():
                        out.append(random.choice(string.ascii_uppercase if ch.isupper() else string.ascii_lowercase))
                    elif ch.isdigit():
                        out.append(random.choice(string.digits))
                    else:
                        out.append(ch)
                return ''.join(out)
            return _full_tokenize(text)

        def tokenize_general(segment: str) -> str:
            out_chars = []
            for ch in segment:
                if ch.isalpha():
                    out_chars.append(
                        random.choice(string.ascii_uppercase if ch.isupper() else string.ascii_lowercase)
                    )
                elif ch.isdigit():
                    out_chars.append(random.choice(string.digits))
                else:
                    # preserve whitespace, punctuation, and other symbols
                    out_chars.append(ch)
            return ''.join(out_chars)

        def tokenize_pan(segment: str) -> str:
            # Keep first 6 and last 4 digits as original; middle digits -> random uppercase letters.
            chars = list(segment)
            digit_positions = [i for i,c in enumerate(chars) if c.isdigit()]
            n = len(digit_positions)
            if n == 0:
                return tokenize_general(segment)
            first_n, last_n = 6, 4
            for idx_pos, pos in enumerate(digit_positions):
                if idx_pos < first_n or idx_pos >= n - last_n:
                    # keep original digit
                    continue
                else:
                    # replace digit with random uppercase letter
                    chars[pos] = random.choice(string.ascii_uppercase)
            return ''.join(chars)

        out = []
        last = 0
        for start, end, etype in merged:
            if last < start:
                out.append(text[last:start])
            segment = text[start:end]
            if (etype and ('CREDIT_CARD' in etype or etype == 'PAN')) or self._is_likely_pan(segment):
                out.append(tokenize_pan(segment))
            else:
                out.append(tokenize_general(segment))
            last = end
        if last < len(text):
            out.append(text[last:])
        return ''.join(out)
    
    def _display_search_results(self, original_text: str, results: Dict, show_anonymized: bool, max_results: int):
        """Display interactive search results"""
        built_in_only = st.session_state.get('__built_in_only__', False)
        # Summary metrics respecting built-in-only mode
        presidio_count = len(results.get('presidio_findings', []))
        transformer_count = len(results.get('transformer_findings', []))
        ollama_count = len(results.get('ollama_findings', []))
        raw_uk = len(results.get('uk_pattern_findings', []))
        uk_count = 0 if built_in_only else raw_uk
        total_count = presidio_count + transformer_count + ollama_count + uk_count
        
        # Display summary
        st.subheader("📊 Search Results Summary")
        if built_in_only:
            st.markdown("<div style='display:inline-block;padding:4px 10px;margin-bottom:6px;background:#0d47a1;color:#fff;border-radius:4px;font-size:12px;font-weight:600;'>BUILT-IN PRESIDIO MODE</div>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Entities", total_count)
        with col2:
            st.metric("Presidio", presidio_count)
        with col3:
            st.metric("Transformers", transformer_count)
        with col4:
            st.metric("Ollama", ollama_count)
        with col5:
            st.metric("UK Patterns", uk_count)
        
        if total_count == 0:
            st.info("✨ No PII entities detected in the text.")
            # Do not render output blocks here to avoid duplication; Interactive Search already shows them.
            return

        # Detailed findings
        st.subheader("🔍 Detailed Findings")
        
        # Combine all findings
        all_findings = []
        for finding in results.get('presidio_findings', [])[:max_results]:
            # Preserve recognizer attribution when present
            src = finding.get('source', 'Presidio')
            all_findings.append({**finding, 'source': src})
        for finding in results.get('transformer_findings', [])[:max_results]:
            # Preserve and surface the specific transformer model if available
            src = finding.get('source', 'Transformers')
            model_name = None
            if isinstance(src, str) and src.startswith('transformer-'):
                model_name = src.split('transformer-', 1)[1]
                pretty_src = f"Transformer ({model_name})"
            else:
                pretty_src = 'Transformers'
            enriched = {**finding, 'source': pretty_src}
            if model_name:
                enriched['model_name'] = model_name
            all_findings.append(enriched)
        for finding in results.get('ollama_findings', [])[:max_results]:
            all_findings.append({**finding, 'source': 'Ollama'})
        if not built_in_only:
            for finding in results.get('uk_pattern_findings', [])[:max_results]:
                all_findings.append({**finding, 'source': 'UK Patterns'})
        
        # Sort by position in text
        all_findings.sort(key=lambda x: x.get('start', 0))
        
        if all_findings:
            # Create a DataFrame for display
            display_data = []
            for i, finding in enumerate(all_findings[:max_results]):
                display_row = {
                    'Entity Type': finding.get('entity_type', 'Unknown'),
                    'Text': finding.get('text', 'N/A'),
                    'Confidence': f"{float(finding.get('confidence', 0)):.3f}",
                    'Position': f"{finding.get('start', 0)}-{finding.get('end', 0)}",
                    'Source': finding.get('source', 'Unknown'),
                    'Recognizer': finding.get('recognizer', ''),
                }
                # Include model column when available (for transformer findings)
                if 'model_name' in finding:
                    display_row['Model'] = finding.get('model_name')
                else:
                    display_row['Model'] = ''
                display_data.append(display_row)
            
            # Display as table
            df = pd.DataFrame(display_data)
            st.dataframe(df, width="stretch", hide_index=True)
            
            # Highlight entities in text
            if st.checkbox("🎯 Highlight Entities in Text"):
                self._display_highlighted_text(original_text, all_findings[:max_results])
        else:
            st.info("No detailed findings to display.")

    def _fallback_anonymize(self, text: str, results: Dict) -> str:
        """Simple masking anonymization when Presidio anonymizer is disabled.

        Replaces detected spans from all enabled sources with <ENTITY_TYPE> tokens.
        Handles overlapping spans by keeping the longest span starting earliest.
        """
        # Collect findings from available sources
        span_sources = [
            ('presidio_findings', True),
            ('transformer_findings', True),
            ('ollama_findings', True),
            ('uk_pattern_findings', not st.session_state.get('__built_in_only__', False))
        ]
        spans = []
        for key, allowed in span_sources:
            if not allowed:
                continue
            for f in results.get(key, []) or []:
                try:
                    start = int(f['start']); end = int(f['end'])
                    etype = str(f.get('entity_type') or f.get('type') or 'PII')
                    seg = text[start:end]
                    if self._is_likely_pan(seg):
                        etype = 'CREDIT_CARD'
                    if 0 <= start < end <= len(text):
                        spans.append((start, end, etype))
                except Exception:
                    continue
        if not spans:
            return text
        # Sort by start then decreasing length to prefer longer overlapping spans
        spans.sort(key=lambda s: (s[0], -(s[1]-s[0])))
        merged = []
        last_end = -1
        for s in spans:
            if s[0] >= last_end:  # non-overlapping
                merged.append(s)
                last_end = s[1]
            else:
                # Overlap: skip shorter span (since sorted by length descending per start)
                continue
        # Build anonymized string
        out = []
        cursor = 0
        for start, end, etype in merged:
            if cursor < start:
                out.append(text[cursor:start])
            tag = etype.upper() if etype else 'PII'
            if self._is_likely_pan(text[start:end]):
                tag = 'CREDIT_CARD'
            out.append(f"<{tag}>")
            cursor = end
        if cursor < len(text):
            out.append(text[cursor:])
        return ''.join(out)
    
    def _display_highlighted_text(self, text: str, findings: List[Dict]):
        """Display text with highlighted PII entities"""
        if not findings:
            st.write(text)
            return
        
        # Sort findings by start position (reverse order for replacement)
        sorted_findings = sorted(findings, key=lambda x: x.get('start', 0), reverse=True)
        
        # Color mapping for different entity types
        color_map = {
            'PERSON': '#FF6B6B',
            'EMAIL_ADDRESS': '#4ECDC4',
            'PHONE_NUMBER': '#45B7D1',
            'DATE_TIME': '#96CEB4',
            'LOCATION': '#FECA57',
            'CREDIT_CARD': '#FF9FF3',
            'UK_POSTCODE': '#54A0FF',
            'UK_NHS_NUMBER': '#5F27CD',
            'ORGANIZATION': '#FF7675',
            'IBAN_CODE': '#74B9FF'
        }
        
        highlighted_text = text
        for finding in sorted_findings:
            start = finding.get('start', 0)
            end = finding.get('end', len(text))
            entity_type = finding.get('entity_type', 'Unknown')
            entity_text = finding.get('text', '')
            confidence = finding.get('confidence', 0)
            source = finding.get('source', 'Unknown')
            
            color = color_map.get(entity_type, '#DDA0DD')
            
            # Create highlighted span
            highlighted_span = f'''<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px; color: white; font-weight: bold;" title="{entity_type} (Confidence: {confidence:.3f}, Source: {source})">{entity_text}</mark>'''
            
            # Replace in text
            highlighted_text = highlighted_text[:start] + highlighted_span + highlighted_text[end:]
        
        st.markdown("**Text with Highlighted PII Entities:**")
        st.markdown(highlighted_text, unsafe_allow_html=True)
        
        # Legend
        st.markdown("**Legend:**")
        legend_cols = st.columns(len(color_map))
        for i, (entity_type, color) in enumerate(color_map.items()):
            with legend_cols[i % len(legend_cols)]:
                st.markdown(f'<span style="background-color: {color}; color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold; font-size: 0.8em;">{entity_type}</span>', unsafe_allow_html=True)
    
    def render_config_page(self):
        """Render the configuration editing page"""
        st.title("⚙️ Configure Analyzers")
        
        tab1, tab2, tab3 = st.tabs(["Analyzer Configuration", "Main Configuration", "No-Code Recognizers"])
        
        with tab1:
            self.render_analyzer_config_editor()
        
        with tab2:
            self.render_main_config_editor()
        
        with tab3:
            self.render_recognizers_config_editor()
    
    def render_analyzer_config_editor(self):
        """Render analyzer configuration editor"""
        st.header("Analyzer Configuration (analyzers_config.yaml)")
        
        if not self.analyzer_config_path.exists():
            st.error("analyzers_config.yaml not found!")
            return
        
        config = self.load_yaml_file(self.analyzer_config_path)
        if not config:
            return
        
        # Entity types configuration
        st.subheader("Entity Types")
        pii_entities = config.get('pii_entities', [])
        
        # Allow editing entity types
        entity_types_text = st.text_area(
            "PII Entity Types (one per line)",
            value='\n'.join(pii_entities),
            height=150,
            help="List of PII entity types to detect"
        )
        
        # Analyzers configuration
        st.subheader("Analyzers")
        analyzers_config = config.get('analyzers', {})
        
        # Create columns for analyzer settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Presidio Analyzer**")
            presidio_enabled = st.checkbox(
                "Enable Presidio",
                value=analyzers_config.get('presidio', {}).get('enabled', True),
                key="presidio_enabled"
            )
            
            presidio_threshold = st.slider(
                "Sensitivity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=analyzers_config.get('presidio', {}).get('sensitivity_threshold', 0.35),
                step=0.05,
                key="presidio_threshold"
            )
            
            st.write("**UK Recognizers**")
            uk_enabled = st.checkbox(
                "Enable UK Recognizers",
                value=analyzers_config.get('uk_recognizers', {}).get('enabled', True),
                key="uk_enabled"
            )
        
        with col2:
            st.write("**Transformer Analyzer**")
            transformer_enabled = st.checkbox(
                "Enable Transformer",
                value=analyzers_config.get('transformer', {}).get('enabled', True),
                key="transformer_enabled"
            )
            
            transformer_model = st.selectbox(
                "Transformer Model",
                options=[
                    "dslim/bert-base-NER",
                    "dbmdz/bert-large-cased-finetuned-conll03-english",
                    "microsoft/DialoGPT-medium"
                ],
                index=0,
                key="transformer_model"
            )
            
            st.write("**Ollama Analyzer**")
            ollama_enabled = st.checkbox(
                "Enable Ollama",
                value=analyzers_config.get('ollama', {}).get('enabled', False),
                key="ollama_enabled"
            )
        
        # Overlap resolution settings
        st.subheader("Overlap Resolution")
        overlap_config = config.get('overlap_resolution', {})
        
        col3, col4 = st.columns(2)
        
        with col3:
            overlap_enabled = st.checkbox(
                "Enable Overlap Resolution",
                value=overlap_config.get('enabled', True),
                key="overlap_enabled"
            )
            
            prioritize_person = st.checkbox(
                "Prioritize PERSON Entities",
                value=overlap_config.get('prioritize_person', True),
                key="prioritize_person"
            )
        
        with col4:
            confidence_threshold_diff = st.slider(
                "Confidence Threshold Difference",
                min_value=0.0,
                max_value=0.5,
                value=overlap_config.get('confidence_threshold_diff', 0.1),
                step=0.01,
                key="confidence_threshold_diff"
            )
        
        # Save button
        if st.button("💾 Save Configuration", type="primary"):
            # Update configuration
            new_config = {
                'pii_entities': [t.strip() for t in entity_types_text.split('\n') if t.strip()],
                'analyzers': {
                    'presidio': {
                        'enabled': presidio_enabled,
                        'description': "Standard Presidio analyzer with built-in recognizers",
                        'sensitivity_threshold': presidio_threshold,
                        'language': 'en'
                    },
                    'uk_recognizers': {
                        'enabled': uk_enabled,
                        'description': "UK-specific recognizers for postcodes, sort codes, etc.",
                        'module': 'uk_recognizers',
                        'function': 'register_uk_recognizers'
                    },
                    'transformer': {
                        'enabled': transformer_enabled,
                        'description': "Transformer-based NER using BERT models",
                        'module': 'transformer_integration',
                        'function': 'get_analyzer_with_transformers',
                        'config': {
                            'sensitivity': 0.35,
                            'uk_specific': True,
                            'transformer_model': transformer_model
                        }
                    },
                    'ollama': {
                        'enabled': ollama_enabled,
                        'description': "Ollama-based entity extraction using local LLMs",
                        'module': 'ollama_integration',
                        'class': 'OllamaEntityExtractor',
                        'config': {
                            'model_name': 'mistral:7b-instruct',
                            'temperature': 0.1
                        }
                    }
                },
                'overlap_resolution': {
                    'enabled': overlap_enabled,
                    'prioritize_person': prioritize_person,
                    'confidence_threshold_diff': confidence_threshold_diff
                },
                'processing': config.get('processing', {
                    'target_field': 'customer_message',
                    'max_records': 1000,
                    'num_threads': 4,
                    'use_threading': True
                })
            }
            
            if self.save_yaml_file(self.analyzer_config_path, new_config):
                st.success("✅ Configuration saved successfully!")
                st.rerun()

        # Backup restore UI
        with st.expander("🕒 Restore Previous Analyzer Config", expanded=False):
            backups = self.list_yaml_backups(self.analyzer_config_path)
            if backups:
                labels = [f"{b.name} (saved {datetime.fromtimestamp(b.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})" for b in backups]
                idx = st.selectbox("Select backup", options=list(range(len(backups))), format_func=lambda i: labels[i], key="analyzer_backup_select")
                selected_backup = backups[idx]
                # Load texts
                try:
                    with open(selected_backup, 'r') as bf:
                        backup_text = bf.read()
                    current_text = ''
                    if self.analyzer_config_path.exists():
                        with open(self.analyzer_config_path, 'r') as cf:
                            current_text = cf.read()
                except Exception as e:
                    st.error(f"Failed to load backup: {e}")
                    backup_text = ''
                    current_text = ''

                diff_col1, diff_col2, diff_col3 = st.columns(3)
                with diff_col1:
                    if st.button("🔍 Show Diff", key="analyzer_show_diff"):
                        diff = difflib.unified_diff(
                            backup_text.splitlines(),
                            current_text.splitlines(),
                            fromfile=selected_backup.name,
                            tofile=self.analyzer_config_path.name,
                            lineterm=''
                        )
                        diff_text = '\n'.join(diff)
                        st.code(diff_text or 'No differences', language='diff')
                with diff_col2:
                    st.download_button(
                        "⬇️ Download Backup",
                        data=backup_text,
                        file_name=selected_backup.name,
                        mime='text/plain',
                        key="analyzer_download_backup"
                    )
                with diff_col3:
                    if st.button("♻️ Restore Backup", key="analyzer_restore_btn"):
                        if self.restore_yaml_backup(self.analyzer_config_path, selected_backup):
                            st.rerun()
            else:
                st.info("No backups yet. A backup is created automatically on each save.")
    
    def render_main_config_editor(self):
        """Render main configuration editor"""
        st.header("Main Configuration (config.yaml)")
        
        if not self.config_path.exists():
            st.warning("config.yaml not found - will create new configuration")
            config = {}
        else:
            config = self.load_yaml_file(self.config_path)
        
        # Basic settings
        st.subheader("Basic Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_records = st.number_input(
                "Default Max Records",
                min_value=1,
                max_value=10000,
                value=config.get('default_max_records', 1000)
            )
            
            default_threads = st.slider(
                "Default Threads",
                min_value=1,
                max_value=8,
                value=config.get('default_threads', 4)
            )
        
        with col2:
            log_level = st.selectbox(
                "Log Level",
                options=["DEBUG", "INFO", "WARNING", "ERROR"],
                index=1,
                help="Logging verbosity level"
            )
        
        # Save button
        if st.button("💾 Save Main Configuration", type="primary"):
            new_config = {
                'default_max_records': default_records,
                'default_threads': default_threads,
                'log_level': log_level
            }
            
            if self.save_yaml_file(self.config_path, new_config):
                st.success("✅ Main configuration saved successfully!")

        with st.expander("🕒 Restore Previous Main Config", expanded=False):
            backups = self.list_yaml_backups(self.config_path)
            if backups:
                labels = [f"{b.name} (saved {datetime.fromtimestamp(b.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})" for b in backups]
                idx = st.selectbox("Select backup", options=list(range(len(backups))), key="main_cfg_restore", format_func=lambda i: labels[i])
                selected_backup = backups[idx]
                try:
                    with open(selected_backup, 'r') as bf:
                        backup_text = bf.read()
                    current_text = ''
                    if self.config_path.exists():
                        with open(self.config_path, 'r') as cf:
                            current_text = cf.read()
                except Exception as e:
                    st.error(f"Failed to load backup: {e}")
                    backup_text = ''
                    current_text = ''

                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("🔍 Show Diff", key="main_show_diff"):
                        diff = difflib.unified_diff(
                            backup_text.splitlines(),
                            current_text.splitlines(),
                            fromfile=selected_backup.name,
                            tofile=self.config_path.name,
                            lineterm=''
                        )
                        diff_text = '\n'.join(diff)
                        st.code(diff_text or 'No differences', language='diff')
                with c2:
                    st.download_button(
                        "⬇️ Download Backup",
                        data=backup_text,
                        file_name=selected_backup.name,
                        mime='text/plain',
                        key="main_download_backup"
                    )
                with c3:
                    if st.button("♻️ Restore Backup", key="main_restore_btn"):
                        if self.restore_yaml_backup(self.config_path, selected_backup):
                            st.rerun()
            else:
                st.info("No backups yet. A backup is created automatically on each save.")
    
    def render_recognizers_config_editor(self):
        """Render no-code recognizers configuration editor"""
        st.header("No-Code Recognizers (presidio_recognizers_config.yaml)")
        
        if not self.recognizers_config_path.exists():
            st.error("❌ presidio_recognizers_config.yaml not found!")
            st.info("💡 This file contains the YAML no-code recognizers configuration")
            if st.button("🔄 Create Template"):
                # Create a basic template
                template = {
                    "recognizers": []
                }
                if self.save_yaml_file(self.recognizers_config_path, template):
                    st.success("✅ Created template recognizers config file")
                    st.rerun()
            return
        
        config = self.load_yaml_file(self.recognizers_config_path)
        if not config:
            return
        
        # Show overview
        recognizers = config.get('recognizers', [])
        st.info(f"📊 Total recognizers defined: **{len(recognizers)}**")
        
        # Tabs for different views
        view_tab, edit_tab, add_tab, help_tab = st.tabs(["👁️ View", "✏️ Edit", "➕ Add New", "❓ Help"])
        
        with view_tab:
            self.render_recognizers_view(recognizers)
        
        with edit_tab:
            self.render_recognizers_edit(config)
        
        with add_tab:
            self.render_recognizers_add()
        
        with help_tab:
            self.render_recognizers_help()
    
    def render_recognizers_view(self, recognizers: List[Dict]):
        """Render recognizers in a readable format"""
        st.subheader("📋 Current Recognizers")
        
        if not recognizers:
            st.warning("No recognizers configured")
            return
        
        # Group by entity type
        entities_dict = {}
        for recognizer in recognizers:
            entity_type = recognizer.get('supported_entity', 'Unknown')
            if entity_type not in entities_dict:
                entities_dict[entity_type] = []
            entities_dict[entity_type].append(recognizer)
        
        # Display by entity type
        for entity_type in sorted(entities_dict.keys()):
            with st.expander(f"🏷️ {entity_type} ({len(entities_dict[entity_type])} recognizers)"):
                for i, recognizer in enumerate(entities_dict[entity_type]):
                    st.markdown(f"**Recognizer {i+1}: {recognizer.get('name', 'Unnamed')}**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text(f"Type: {recognizer.get('type', 'Unknown')}")
                        st.text(f"Language: {recognizer.get('supported_language', 'en')}")
                        st.text(f"Score: {recognizer.get('score', 'N/A')}")
                    
                    with col2:
                        patterns = recognizer.get('patterns', [])
                        if patterns:
                            st.text("Patterns:")
                            for pattern in patterns:
                                pattern_name = pattern.get('name', 'Unnamed')
                                pattern_regex = pattern.get('regex', 'N/A')
                                st.code(f"{pattern_name}: {pattern_regex[:50]}...")
                    
                    context = recognizer.get('context', [])
                    if context:
                        st.text("Context keywords:")
                        st.write(", ".join(context))
                    
                    st.divider()
    
    def render_recognizers_edit(self, config: Dict):
        """Render recognizers editor as YAML text"""
        st.subheader("✏️ Edit YAML Configuration")
        
        st.warning("⚠️ Advanced editing - Make sure YAML syntax is correct!")
        
        # Convert config to YAML string for editing
        try:
            yaml_content = yaml.dump(config, default_flow_style=False, indent=2, sort_keys=False)
        except Exception as e:
            st.error(f"Error converting config to YAML: {e}")
            return
        
        # Text editor
        edited_yaml = st.text_area(
            "YAML Configuration",
            value=yaml_content,
            height=600,
            help="Edit the YAML configuration directly. Be careful with indentation!"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("💾 Save Changes", type="primary"):
                try:
                    # Parse the edited YAML
                    new_config = yaml.safe_load(edited_yaml)
                    
                    # Save to file
                    if self.save_yaml_file(self.recognizers_config_path, new_config):
                        st.success("✅ Configuration saved successfully!")
                        st.rerun()
                except yaml.YAMLError as e:
                    st.error(f"❌ YAML syntax error: {e}")
                except Exception as e:
                    st.error(f"❌ Error saving configuration: {e}")
        
        with col2:
            if st.button("🔄 Reset to Original"):
                st.rerun()
        
        with col3:
            if st.button("✅ Validate YAML"):
                try:
                    yaml.safe_load(edited_yaml)
                    st.success("✅ YAML syntax is valid!")
                except yaml.YAMLError as e:
                    st.error(f"❌ YAML syntax error: {e}")

        with st.expander("🕒 Restore Previous Recognizers Config", expanded=False):
            backups = self.list_yaml_backups(self.recognizers_config_path)
            if backups:
                labels = [f"{b.name} (saved {datetime.fromtimestamp(b.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})" for b in backups]
                idx = st.selectbox("Select backup", options=list(range(len(backups))), key="rec_cfg_restore", format_func=lambda i: labels[i])
                selected_backup = backups[idx]
                try:
                    with open(selected_backup, 'r') as bf:
                        backup_text = bf.read()
                    current_text = ''
                    if self.recognizers_config_path.exists():
                        with open(self.recognizers_config_path, 'r') as cf:
                            current_text = cf.read()
                except Exception as e:
                    st.error(f"Failed to load backup: {e}")
                    backup_text = ''
                    current_text = ''

                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    if st.button("🔍 Show Diff", key="rec_show_diff"):
                        diff = difflib.unified_diff(
                            backup_text.splitlines(),
                            current_text.splitlines(),
                            fromfile=selected_backup.name,
                            tofile=self.recognizers_config_path.name,
                            lineterm=''
                        )
                        diff_text = '\n'.join(diff)
                        st.code(diff_text or 'No differences', language='diff')
                with rc2:
                    st.download_button(
                        "⬇️ Download Backup",
                        data=backup_text,
                        file_name=selected_backup.name,
                        mime='text/plain',
                        key="rec_download_backup"
                    )
                with rc3:
                    if st.button("♻️ Restore Backup", key="rec_restore_btn"):
                        if self.restore_yaml_backup(self.recognizers_config_path, selected_backup):
                            st.rerun()
            else:
                st.info("No backups yet. A backup is created automatically on each save.")
    
    def render_recognizers_add(self):
        """Render interface to add new recognizers"""
        st.subheader("➕ Add New Recognizer")
        
        # Basic recognizer template
        recognizer_templates = {
            "Pattern Recognizer": {
                "name": "NEW_RECOGNIZER",
                "type": "PatternRecognizer", 
                "supported_entity": "CUSTOM_ENTITY",
                "supported_language": "en",
                "score": 1.0,
                "patterns": [
                    {
                        "name": "pattern_1",
                        "regex": "YOUR_REGEX_HERE",
                        "score": 1.0
                    }
                ],
                "context": ["keyword1", "keyword2"]
            }
        }
        
        template_type = st.selectbox(
            "Select Template",
            options=list(recognizer_templates.keys())
        )
        
        if template_type:
            template = recognizer_templates[template_type]
            
            st.subheader("📝 Configure New Recognizer")
            
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Name", value=template["name"])
                entity_type = st.text_input("Entity Type", value=template["supported_entity"])
                score = st.number_input("Score", min_value=0.0, max_value=1.0, value=float(template["score"]), step=0.1)
            
            with col2:
                language = st.text_input("Language", value=template["supported_language"])
                recognizer_type = st.text_input("Type", value=template["type"])
            
            # Pattern configuration
            st.subheader("🔍 Patterns")
            pattern_name = st.text_input("Pattern Name", value="pattern_1")
            pattern_regex = st.text_area(
                "Regular Expression",
                value="YOUR_REGEX_HERE",
                help="Enter the regular expression for pattern matching"
            )
            
            # Context keywords
            st.subheader("📝 Context Keywords")
            context_keywords = st.text_input(
                "Context Keywords (comma-separated)",
                value="keyword1, keyword2",
                help="Optional context keywords to improve accuracy"
            )
            
            if st.button("➕ Add Recognizer", type="primary"):
                if name and entity_type and pattern_regex != "YOUR_REGEX_HERE":
                    new_recognizer = {
                        "name": name,
                        "type": recognizer_type,
                        "supported_entity": entity_type,
                        "supported_language": language,
                        "score": score,
                        "patterns": [
                            {
                                "name": pattern_name,
                                "regex": pattern_regex,
                                "score": score
                            }
                        ]
                    }
                    
                    if context_keywords.strip():
                        keywords = [k.strip() for k in context_keywords.split(",") if k.strip()]
                        new_recognizer["context"] = keywords
                    
                    # Load current config and add new recognizer
                    config = self.load_yaml_file(self.recognizers_config_path)
                    if 'recognizers' not in config:
                        config['recognizers'] = []
                    
                    config['recognizers'].append(new_recognizer)
                    
                    if self.save_yaml_file(self.recognizers_config_path, config):
                        st.success(f"✅ Added recognizer '{name}' successfully!")
                        st.rerun()
                else:
                    st.error("❌ Please fill in all required fields and provide a valid regex pattern")
    
    def render_recognizers_help(self):
        """Render help information for recognizers configuration"""
        st.subheader("❓ Help & Documentation")
        
        st.markdown("""
        ### 🎯 What are No-Code Recognizers?
        
        No-code recognizers allow you to define PII detection patterns using YAML configuration files 
        instead of writing Python code. This makes it easier to:
        
        - ✅ Add new entity types without coding
        - ✅ Modify detection patterns quickly  
        - ✅ Version control your recognition rules
        - ✅ Share configurations across teams
        - ✅ A/B test different detection approaches
        
        ### 📋 Recognizer Structure
        
        Each recognizer has these components:
        
        ```yaml
        - name: UK_POSTCODE_RECOGNIZER
          type: PatternRecognizer
          supported_entity: UK_POSTCODE
          supported_language: en
          score: 1.0
          patterns:
            - name: uk_postcode_pattern
              regex: "\\b[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}\\b"
              score: 1.0
          context: ["postcode", "postal code", "zip"]
        ```
        
        ### 🔧 Configuration Fields
        
        - **name**: Unique identifier for the recognizer
        - **type**: Usually "PatternRecognizer" for regex-based detection
        - **supported_entity**: The PII entity type this recognizer detects
        - **supported_language**: Language code (usually "en")
        - **score**: Confidence score (0.0 to 1.0)
        - **patterns**: List of regex patterns to match
        - **context**: Optional keywords that improve detection accuracy
        
        ### 🎨 Entity Types
        
        Your system supports these entity types:
        - `UK_POSTCODE` - UK postal codes
        - `UK_SORT_CODE` - UK bank sort codes  
        - `UK_NHS_NUMBER` - NHS numbers
        - `UK_NI_NUMBER` - National Insurance numbers
        - `UK_BANK_ACCOUNT` - Bank account numbers
        - `UK_MONETARY_VALUE` - UK currency amounts
        - `UK_PASSPORT` - UK passport numbers
        - `UK_PHONE_NUMBER` - UK phone numbers
        - `UK_VAT_NUMBER` - VAT registration numbers
        - `UK_COMPANY_NUMBER` - Company registration numbers
        - `UK_VEHICLE_REGISTRATION` - Vehicle registration plates
        - `UK_DRIVING_LICENSE` - Driving license numbers
        - `CUSTOMER_ID` - Customer ID patterns
        - `ENHANCED_EMAIL` - Enhanced email detection
        - `ENHANCED_CREDIT_CARD` - Enhanced credit card detection
        - `ENHANCED_IBAN` - Enhanced IBAN detection
        
        ### 🧪 Testing Your Recognizers
        
        After making changes, you can test them using:
        
        ```bash
        python no_code_demo.py
        python test_yaml_integration.py
        ```
        
        ### 💡 Tips for Success
        
        1. **Test your regex patterns** before adding them
        2. **Use appropriate confidence scores** (0.7-1.0 for high confidence)
        3. **Add context keywords** to reduce false positives
        4. **Make names descriptive** and follow naming conventions
        5. **Backup your configuration** before major changes
        """)
        
        # Quick test tool
        st.subheader("🧪 Quick Pattern Test")
        
        test_pattern = st.text_input(
            "Test Regex Pattern",
            placeholder="\\b[A-Z]{2}[0-9]{2} [A-Z]{3}\\b",
            help="Enter a regex pattern to test"
        )
        
        test_text = st.text_area(
            "Test Text",
            placeholder="Enter text to test the pattern against...",
            height=100
        )
        
        if st.button("🔍 Test Pattern") and test_pattern and test_text:
            try:
                import re
                matches = re.findall(test_pattern, test_text, re.IGNORECASE)
                
                if matches:
                    st.success(f"✅ Found {len(matches)} matches:")
                    for i, match in enumerate(matches, 1):
                        st.code(f"Match {i}: {match}")
                else:
                    st.info("ℹ️ No matches found")
                    
            except re.error as e:
                st.error(f"❌ Invalid regex pattern: {e}")
    
    def render_results_page(self):
        """Render results viewing and analysis page"""
        st.title("📊 View Analysis Results")
        
        json_files = self.get_json_results_files()
        
        if not json_files:
            st.warning("No analysis results found. Run an analysis first.")
            return
        
        # File selection
        selected_file = st.selectbox(
            "Select Results File",
            options=[f.name for f in json_files],
            help="Choose which analysis results to view"
        )
        
        if selected_file:
            file_path = self.results_path / selected_file
            results = self.load_json_results(file_path)
            
            if not results:
                return
            
            # Add global search functionality
            with st.expander("🔍 Global Search", expanded=False):
                st.write("Search across all records and entities in this analysis")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    global_search = st.text_input(
                        "Search term",
                        placeholder="Search across all content, entities, and types...",
                        help="This will search through original text, anonymized text, entity values, and entity types",
                        key="global_search"
                    )
                with col2:
                    global_case_sensitive = st.checkbox("Case sensitive", key="global_case_sensitive")
                
                if global_search:
                    # Search through all records
                    matching_records = []
                    matching_entities = []
                    
                    records = results.get('results', [])
                    for record_idx, record in enumerate(records):
                        record_match = False
                        fields = record.get('fields', {})
                        
                        for field_name, field_data in fields.items():
                            if isinstance(field_data, dict):
                                # Search in original and anonymized text
                                original_text = field_data.get('original_text', '')
                                anonymized_text = field_data.get('anonymized_text', '')
                                
                                search_texts = [original_text, anonymized_text]
                                
                                for text in search_texts:
                                    if global_case_sensitive:
                                        if global_search in text:
                                            record_match = True
                                            break
                                    else:
                                        if global_search.lower() in text.lower():
                                            record_match = True
                                            break
                                
                                # Search in entities
                                entities = field_data.get('entities', [])
                                for entity in entities:
                                    entity_text = entity.get('text', '')
                                    entity_type = entity.get('entity_type', '')
                                    
                                    entity_search_texts = [entity_text, entity_type]
                                    
                                    for text in entity_search_texts:
                                        entity_matches = False
                                        if global_case_sensitive:
                                            if global_search in text:
                                                entity_matches = True
                                        else:
                                            if global_search.lower() in text.lower():
                                                entity_matches = True
                                        
                                        if entity_matches:
                                            record_match = True
                                            matching_entities.append({
                                                'record_id': record.get('record_id', f'Record {record_idx}'),
                                                'field': field_name,
                                                'entity_type': entity_type,
                                                'entity_text': entity_text,
                                                'confidence': entity.get('confidence', 'N/A')
                                            })
                                
                                if record_match:
                                    break
                        
                        if record_match:
                            matching_records.append({
                                'record_id': record.get('record_id', f'Record {record_idx}'),
                                'conversation_id': record.get('conversation_id', 'N/A'),
                                'fields': list(fields.keys())
                            })
                    
                    # Display search results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Matching Records", len(matching_records))
                    with col2:
                        st.metric("Matching Entities", len(matching_entities))
                    
                    if matching_records:
                        with st.expander(f"📄 Matching Records ({len(matching_records)})", expanded=True):
                            for record in matching_records[:10]:  # Show first 10
                                st.write(f"**{record['record_id']}** (Fields: {', '.join(record['fields'])})")
                            if len(matching_records) > 10:
                                st.write(f"... and {len(matching_records) - 10} more records")
                    
                    if matching_entities:
                        with st.expander(f"🏷️ Matching Entities ({len(matching_entities)})", expanded=True):
                            entity_df = pd.DataFrame(matching_entities[:20])  # Show first 20
                            st.dataframe(
                                entity_df,
                                column_config={
                                    "record_id": "Record ID",
                                    "field": "Field",
                                    "entity_type": "Entity Type", 
                                    "entity_text": "Entity Text",
                                    "confidence": "Confidence"
                                },
                                width="stretch",
                                hide_index=True
                            )
                            if len(matching_entities) > 20:
                                st.write(f"... and {len(matching_entities) - 20} more entities")
            
            # Display summary metrics
            self.render_results_summary(results)
            
            # Display detailed analysis
            tab1, tab2, tab3, tab4 = st.tabs([
                "Summary", "Entity Analysis", "Records Detail", "Raw Data"
            ])
            
            with tab1:
                self.render_results_overview(results)
            
            with tab2:
                self.render_entity_analysis(results)
            
            with tab3:
                self.render_records_detail(results)
            
            with tab4:
                self.render_raw_data(results)
    
    def render_results_summary(self, results: Dict):
        """Render results summary metrics"""
        metadata = results.get('metadata', {})
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Records Processed",
                metadata.get('records_processed', 0)
            )
        
        with col2:
            st.metric(
                "Entities Found",
                metadata.get('total_entities_found', 0)
            )
        
        with col3:
            total_time = metadata.get('total_time_seconds', 0)
            st.metric(
                "Total Time",
                f"{total_time:.2f}s"
            )
        
        with col4:
            processing_time = metadata.get('processing_time_seconds', 0)
            st.metric(
                "Processing Time",
                f"{processing_time:.2f}s"
            )
    
    def render_results_overview(self, results: Dict):
        """Render results overview with charts"""
        st.header("Analysis Overview")
        
        metadata = results.get('metadata', {})
        records = results.get('results', [])
        
        # Calculate entity type distribution from actual data
        entity_type_counts = {}
        all_entities = []
        
        for record in records:
            fields = record.get('fields', {})
            for field_name, field_data in fields.items():
                entities = field_data.get('entities', []) if isinstance(field_data, dict) else []
                for entity in entities:
                    entity_type = entity.get('entity_type', 'Unknown')
                    entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
                    all_entities.append(entity)
        
        # Entity type distribution
        if entity_type_counts and PLOTLY_AVAILABLE:
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of entity types
                fig_pie = px.pie(
                    values=list(entity_type_counts.values()),
                    names=list(entity_type_counts.keys()),
                    title="Entity Type Distribution"
                )
                st.plotly_chart(fig_pie, width='stretch')
            
            with col2:
                # Bar chart of entity counts
                fig_bar = px.bar(
                    x=list(entity_type_counts.keys()),
                    y=list(entity_type_counts.values()),
                    title="Entity Counts by Type"
                )
                fig_bar.update_layout(
                    xaxis_title="Entity Type",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_bar, width='stretch')
        elif entity_type_counts:
            # Fallback to simple display without charts
            st.subheader("Entity Type Distribution")
            for entity_type, count in entity_type_counts.items():
                st.write(f"**{entity_type}**: {count}")
        else:
            st.info("No entities found in the analysis results.")
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Timing Information**")
            st.write(f"Initialization: {metadata.get('initialization_time_seconds', 0):.2f}s")
            st.write(f"Processing: {metadata.get('processing_time_seconds', 0):.2f}s")
            st.write(f"Total: {metadata.get('total_time_seconds', 0):.2f}s")
        
        with col2:
            st.write("**Analysis Configuration**")
            st.write(f"Processing mode: {metadata.get('processing_mode', 'Unknown')}")
            st.write(f"Workers: {metadata.get('max_workers', 'Unknown')}")
            config = metadata.get('configuration', {})
            st.write(f"Target field: {config.get('target_field', 'Unknown')}")
            st.write(f"Use Ollama: {metadata.get('use_ollama', False)}")
            st.write(f"Resolve overlaps: {metadata.get('resolve_overlaps', False)}")
            st.write(f"Built-in only mode: {metadata.get('builtin_only', False)}")
    
        # Render recognizer analysis section
        self.render_recognizer_analysis(results)
    
    def render_recognizer_analysis(self, results: Dict):
        """Render detailed recognizer analysis section"""
        metadata = results.get('metadata', {})
        recognizer_info = metadata.get('recognizer_analysis', {})
        
        if not recognizer_info:
            return
            
        st.subheader("🔍 Recognizer Analysis")
        
        # Summary metrics
        col1, col2 = st.columns(2)
        with col1:
            total_recognizers = recognizer_info.get('total_recognizers', 0)
            st.metric("Total Recognizers", total_recognizers)
            
        with col2:
            builtin_only = metadata.get('builtin_only', False)
            mode = "Built-in Only" if builtin_only else "All Recognizers"
            st.metric("Analysis Mode", mode)
        
        # Recognizer categories
        recognizer_categories = recognizer_info.get('recognizer_categories', {})
        
        if recognizer_categories:
            # Create tabs for different recognizer types
            tab_names = []
            tab_data = []
            
            for category, recognizers in recognizer_categories.items():
                if recognizers:  # Only show non-empty categories
                    category_name = category.replace('_', ' ').title()
                    tab_names.append(f"{category_name} ({len(recognizers)})")
                    tab_data.append((category, recognizers))
            
            if tab_names:
                tabs = st.tabs(tab_names)
                
                for i, (tab, (category, recognizers)) in enumerate(zip(tabs, tab_data)):
                    with tab:
                        st.write(f"**{category.replace('_', ' ').title()} Recognizers**")
                        
                        # Create a DataFrame for better display
                        recognizer_data = []
                        for recognizer in recognizers:
                            # Handle both string and dict cases
                            if isinstance(recognizer, str):
                                recognizer_data.append({
                                    'Recognizer': recognizer,
                                    'Supported Entities': 'Unknown',
                                    'Type': 'Unknown'
                                })
                            elif isinstance(recognizer, dict):
                                recognizer_data.append({
                                    'Recognizer': recognizer.get('name', 'Unknown'),
                                    'Supported Entities': ', '.join(recognizer.get('supported_entities', [])),
                                    'Type': recognizer.get('type', 'Unknown')
                                })
                            else:
                                # Handle other unexpected types
                                recognizer_data.append({
                                    'Recognizer': str(recognizer),
                                    'Supported Entities': 'Unknown',
                                    'Type': 'Unknown'
                                })
                        
                        if recognizer_data:
                            st.dataframe(
                                recognizer_data,
                                width="stretch",
                                hide_index=True
                            )
        
        # Entity type mappings
        entity_mappings = recognizer_info.get('entity_type_mappings', {})
        if entity_mappings:
            with st.expander("📋 Entity Type → Recognizer Mappings"):
                st.write("**How each entity type is detected:**")
                
                for entity_type, recognizers in entity_mappings.items():
                    st.write(f"**{entity_type}**")
                    for recognizer in recognizers:
                        # Handle both string and dict cases
                        if isinstance(recognizer, str):
                            st.write(f"  • {recognizer}")
                        elif isinstance(recognizer, dict):
                            recognizer_name = recognizer.get('recognizer', 'Unknown')
                            recognizer_type = recognizer.get('type', 'Unknown')
                            st.write(f"  • {recognizer_name} ({recognizer_type})")
                        else:
                            st.write(f"  • {str(recognizer)}")
                    st.write("")
    
    def render_entity_analysis(self, results: Dict):
        """Render detailed entity analysis"""
        st.header("Entity Analysis")
        
        records = results.get('results', [])
        if not records:
            st.warning("No records found in results")
            return
        
        # Flatten all entities for analysis
        all_entities = []
        for record in records:
            fields = record.get('fields', {})
            for field_name, field_data in fields.items():
                if not isinstance(field_data, dict):
                    continue
                entities = field_data.get('entities', [])
                for entity in entities:
                    entity_info = entity.copy()
                    entity_info['record_id'] = record.get('record_id', 'Unknown')
                    entity_info['field_name'] = field_name
                    entity_info['original_text'] = field_data.get('original_text', '')
                    entity_info['anonymized_text'] = field_data.get('anonymized_text', '')
                    all_entities.append(entity_info)
        
        if not all_entities:
            st.info("No PII entities were detected in this analysis.")
            
            # Show some sample records to help understand why no entities were found
            st.subheader("Sample Record Data")
            sample_records = records[:3]
            for i, record in enumerate(sample_records):
                with st.expander(f"Sample Record {i+1} - {record.get('record_id', 'Unknown')}"):
                    fields = record.get('fields', {})
                    for field_name, field_data in fields.items():
                        if isinstance(field_data, dict):
                            original_text = field_data.get('original_text', 'No text')
                            st.write(f"**{field_name}**: {original_text}")
            
            # Show configuration info
            metadata = results.get('metadata', {})
            config = metadata.get('configuration', {})
            st.subheader("Analysis Configuration")
            st.write(f"**Target field**: {config.get('target_field', 'Unknown')}")
            st.write(f"**PII entities to detect**: {config.get('pii_entities', [])}")
            st.write(f"**Use Ollama**: {metadata.get('use_ollama', False)}")
            
            return
        
        # Add search functionality for entities
        st.subheader("🔍 Search Entities")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            entity_search_term = st.text_input(
                "Search detected entities",
                placeholder="Search entity text, type, or surrounding content...",
                help="Search through detected entities and their context"
            )
        
        with col2:
            entity_search_type = st.selectbox(
                "Search in:",
                ["All", "Entity Text", "Entity Type", "Context"],
                help="Choose what to search in"
            )
        
        with col3:
            entity_case_sensitive = st.checkbox("Case sensitive", value=False, key="entity_search_case")
        
        # Filter entities based on search
        filtered_entities = all_entities
        
        if entity_search_term:
            filtered_entities = []
            
            for entity in all_entities:
                found_match = False
                entity_text = entity.get('text', '')
                entity_type = entity.get('entity_type', '')
                original_context = entity.get('original_text', '')
                anonymized_context = entity.get('anonymized_text', '')
                
                # Prepare search texts based on search type
                search_texts = []
                if entity_search_type in ["All", "Entity Text"]:
                    search_texts.append(entity_text)
                if entity_search_type in ["All", "Entity Type"]:
                    search_texts.append(entity_type)
                if entity_search_type in ["All", "Context"]:
                    search_texts.extend([original_context, anonymized_context])
                
                # Perform search
                for text in search_texts:
                    if entity_case_sensitive:
                        if entity_search_term in text:
                            found_match = True
                            break
                    else:
                        if entity_search_term.lower() in text.lower():
                            found_match = True
                            break
                
                if found_match:
                    filtered_entities.append(entity)
        
        # Show search results
        if entity_search_term:
            st.info(f"Found {len(filtered_entities)} entit(ies) containing '{entity_search_term}'")
        
        # Check if we have entities to display
        if not filtered_entities:
            if entity_search_term:
                st.warning("No entities match your search criteria")
            else:
                st.info("No entities found")
            return
        
        # Convert to DataFrame for analysis
        import pandas as pd
        df_entities = pd.DataFrame(filtered_entities)
        
        # Entity confidence distribution
        if 'confidence' in df_entities.columns and PLOTLY_AVAILABLE:
            try:
                # Ensure confidence column is numeric for plotting
                df_plot = df_entities.copy()
                df_plot['confidence'] = pd.to_numeric(df_plot['confidence'], errors='coerce')
                df_plot = df_plot.dropna(subset=['confidence'])  # Remove invalid confidence values
                
                if not df_plot.empty:
                    fig_conf = px.histogram(
                        df_plot,
                        x='confidence',
                        nbins=20,
                        title="Entity Confidence Distribution"
                    )
                    fig_conf.update_layout(
                        xaxis_title="Confidence Score",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig_conf, width='stretch')
                else:
                    st.write("No valid confidence values for plotting")
            except Exception as e:
                st.write(f"Could not generate confidence histogram: {str(e)}")
        elif 'confidence' in df_entities.columns:
            # Fallback to simple stats
            st.subheader("Confidence Statistics")
            try:
                # Convert confidence column to numeric, handling any string values
                confidence_numeric = pd.to_numeric(df_entities['confidence'], errors='coerce')
                st.write(f"Mean: {confidence_numeric.mean():.3f}")
                st.write(f"Median: {confidence_numeric.median():.3f}")
                st.write(f"Min: {confidence_numeric.min():.3f}")
                st.write(f"Max: {confidence_numeric.max():.3f}")
            except Exception as e:
                st.write("Confidence statistics unavailable")
                st.write(f"Error: {str(e)}")
        
        # Top entities by type
        if 'entity_type' in df_entities.columns and 'text' in df_entities.columns:
            st.subheader("Most Common Entities")
            
            # Group by entity type and text
            entity_counts = df_entities.groupby(['entity_type', 'text']).size().reset_index(name='count')
            entity_counts = entity_counts.sort_values('count', ascending=False).head(20)
            
            st.dataframe(
                entity_counts,
                column_config={
                    "entity_type": "Entity Type",
                    "text": "Text",
                    "count": "Occurrences"
                },
                width='stretch'
            )
    
    def render_records_detail(self, results: Dict):
        """Render detailed records view"""
        st.header("Records Detail")
        
        records = results.get('results', [])
        if not records:
            st.warning("No records found in results")
            return
        
        # Search functionality
        st.subheader("🔍 Search Records")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_term = st.text_input(
                "Search for text in records",
                placeholder="Enter search term...",
                help="Search will look through both original and anonymized content"
            )
        
        with col2:
            search_type = st.selectbox(
                "Search in:",
                ["Both", "Original Only", "Anonymized Only"],
                help="Choose which content to search"
            )
        
        with col3:
            case_sensitive = st.checkbox("Case sensitive", value=False)
        
        # Filter records based on search
        filtered_records = records
        filtered_indices = list(range(len(records)))
        
        if search_term:
            filtered_records = []
            filtered_indices = []
            
            for idx, record in enumerate(records):
                fields = record.get('fields', {})
                found_match = False
                
                for field_name, field_data in fields.items():
                    if isinstance(field_data, dict):
                        original_text = field_data.get('original_text', '')
                        anonymized_text = field_data.get('anonymized_text', '')
                        
                        # Prepare search text based on search type
                        search_texts = []
                        if search_type in ["Both", "Original Only"]:
                            search_texts.append(original_text)
                        if search_type in ["Both", "Anonymized Only"]:
                            search_texts.append(anonymized_text)
                        
                        # Perform search
                        for text in search_texts:
                            if case_sensitive:
                                if search_term in text:
                                    found_match = True
                                    break
                            else:
                                if search_term.lower() in text.lower():
                                    found_match = True
                                    break
                        
                        if found_match:
                            break
                
                if found_match:
                    filtered_records.append(record)
                    filtered_indices.append(idx)
        
        # Show search results
        if search_term:
            st.info(f"Found {len(filtered_records)} record(s) containing '{search_term}'")
        
        if not filtered_records:
            if search_term:
                st.warning("No records match your search criteria")
            return
        
        # Record selection
        record_options = [f"Record {r.get('record_id', filtered_indices[i])}" for i, r in enumerate(filtered_records)]
        selected_record_idx = st.selectbox(
            "Select Record",
            range(len(record_options)),
            format_func=lambda x: record_options[x]
        )
        
        if selected_record_idx is not None:
            record = filtered_records[selected_record_idx]
            original_idx = filtered_indices[selected_record_idx]
            
            # Highlight search terms in the display
            def highlight_search_term(text, term, case_sensitive=False):
                if not term:
                    return text
                
                if case_sensitive:
                    return text.replace(term, f"🔍**{term}**🔍")
                else:
                    # Case insensitive highlighting
                    import re
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    return pattern.sub(lambda m: f"🔍**{m.group()}**🔍", text)
            
            # Display record details
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Fields and Content")
                fields = record.get('fields', {})
                
                for field_name, field_data in fields.items():
                    if isinstance(field_data, dict):
                        original_text = field_data.get('original_text', '')
                        anonymized_text = field_data.get('anonymized_text', '')
                        entity_count = field_data.get('entity_count', 0)
                        
                        st.write(f"**Field: {field_name}** ({entity_count} entities)")
                        
                        # Original text with highlighting
                        display_original = highlight_search_term(original_text, search_term, case_sensitive) if search_term else original_text
                        st.text_area(
                            f"Original Content - {field_name}",
                            value=display_original,
                            height=80,
                            disabled=True,
                            key=f"field_original_{field_name}_{original_idx}"
                        )
                        
                        # Anonymized text (if available) with highlighting
                        if anonymized_text:
                            display_anonymized = highlight_search_term(anonymized_text, search_term, case_sensitive) if search_term else anonymized_text
                            st.text_area(
                                f"Anonymized Content - {field_name}",
                                value=display_anonymized,
                                height=80,
                                disabled=True,
                                key=f"field_anonymized_{field_name}_{original_idx}",
                                help="Text with PII entities replaced with placeholders"
                            )
                        else:
                            st.info(f"No anonymized text available for {field_name}")
                        
                        st.markdown("---")
            
            with col2:
                st.subheader("Record Info")
                st.write(f"**Record ID:** {record.get('record_id', 'N/A')}")
                st.write(f"**Conversation ID:** {record.get('conversation_id', 'N/A')}")
                st.write(f"**Customer ID:** {record.get('customer_id', 'N/A')}")
                
                # Count total entities across all fields
                total_entities = 0
                fields = record.get('fields', {})
                for field_data in fields.values():
                    if isinstance(field_data, dict):
                        total_entities += field_data.get('entity_count', 0)
                st.write(f"**Total Entities Found:** {total_entities}")
                
                # Ground truth info if available
                ground_truth = record.get('ground_truth_pii', {})
                if ground_truth:
                    st.write(f"**Ground Truth Count:** {ground_truth.get('count', 0)}")
            
            # Display entities from all fields
            all_entities = []
            fields = record.get('fields', {})
            for field_name, field_data in fields.items():
                if isinstance(field_data, dict):
                    entities = field_data.get('entities', [])
                    for entity in entities:
                        entity_copy = entity.copy()
                        entity_copy['field'] = field_name
                        all_entities.append(entity_copy)
            
            if all_entities:
                st.subheader("Detected Entities")
                
                entity_data = []
                for entity in all_entities:
                    confidence = entity.get('confidence', 0)
                    # Safely convert confidence to float for formatting
                    try:
                        confidence_float = float(confidence)
                        confidence_str = f"{confidence_float:.3f}"
                    except (ValueError, TypeError):
                        confidence_str = str(confidence)
                    
                    entity_data.append({
                        'Field': entity.get('field', ''),
                        'Text': entity.get('text', ''),
                        'Type': entity.get('entity_type', ''),
                        'Confidence': confidence_str,
                        'Start': entity.get('start', 0),
                        'End': entity.get('end', 0),
                        'Source': entity.get('source', 'Unknown')
                    })
                
                import pandas as pd
                df_entities = pd.DataFrame(entity_data)
                st.dataframe(df_entities, width='stretch')
                
                # Add anonymized text comparison section
                st.subheader("Anonymization Results")
                
                fields = record.get('fields', {})
                for field_name, field_data in fields.items():
                    if isinstance(field_data, dict):
                        original_text = field_data.get('original_text', '')
                        anonymized_text = field_data.get('anonymized_text', '')
                        entity_count = field_data.get('entity_count', 0)
                        
                        if entity_count > 0 and anonymized_text:
                            with st.expander(f"Before/After Comparison: {field_name} ({entity_count} entities anonymized)"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**🔴 Original (with PII):**")
                                    st.code(original_text, language=None)
                                    
                                with col2:
                                    st.write("**🔒 Anonymized (PII removed):**")
                                    st.code(anonymized_text, language=None)
                        elif entity_count == 0:
                            st.info(f"Field '{field_name}': No PII detected - text unchanged")
                        else:
                            st.warning(f"Field '{field_name}': Anonymized text not available")
            else:
                st.info("No entities detected in this record")
    
    def render_raw_data(self, results: Dict):
        """Render raw JSON data"""
        st.header("Raw Data")
        
        # Display formatted JSON
        st.json(results)
        
        # Download button
        json_str = json.dumps(results, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"pii_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def render_exposure_analysis_page(self):
        """Render PII exposure analysis page"""
        st.title("🔎 PII Exposure Analysis")
        
        st.markdown("""
        Analyze PII detection results to identify potential data exposure in anonymized outputs.
        This tool checks if any original PII data is still visible in the anonymized text.
        """)
        
        # File selection
        st.header("Select Results File")
        json_files = self.get_json_results_files()
        
        if not json_files:
            st.warning("No result files found in the results/ folder. Run an analysis first.")
            return
        
        file_options = [f.name for f in json_files]
        selected_file = st.selectbox("Choose a results file to analyze:", file_options)
        
        if not selected_file:
            return
        
        selected_path = self.results_path / selected_file
        
        # Analysis controls
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("🔍 Run Exposure Analysis", type="primary"):
                st.session_state.exposure_analysis_running = True
        
        with col2:
            st.info("This analysis will examine the selected results file for potential PII exposure.")
        
        # Run analysis if requested
        if getattr(st.session_state, 'exposure_analysis_running', False):
            try:
                with st.spinner("Analyzing PII exposure..."):
                    # Initialize analyzer
                    analyzer = PIIExposureAnalyzer()
                    
                    # Load results file
                    results_data = analyzer.load_results_file(str(selected_path))
                    
                    if not results_data:
                        st.error("Could not load data from the selected file.")
                        st.session_state.exposure_analysis_running = False
                        return
                    
                    st.info(f"Loaded {len(results_data)} records for analysis...")
                    
                    # Run analysis with manual progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Initialize summary structure
                    summary = {
                        'total_records': len(results_data),
                        'no_exposure': 0,
                        'partial_exposure': 0,
                        'full_exposure': 0,
                        'exposure_details': [],
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                    
                    # Analyze records with progress updates
                    for i, record in enumerate(results_data):
                        if i % 50 == 0:  # Update every 50 records
                            progress = int((i / len(results_data)) * 100)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing record {i+1}/{len(results_data)}")
                        
                        try:
                            analysis = analyzer.analyze_record(record)
                            summary['exposure_details'].append(analysis)
                            
                            # Update counters
                            exposure_level = analysis['overall_exposure_level']
                            if exposure_level == 'NO_EXPOSURE':
                                summary['no_exposure'] += 1
                            elif exposure_level == 'PARTIAL_EXPOSURE':
                                summary['partial_exposure'] += 1
                            elif exposure_level == 'FULL_EXPOSURE':
                                summary['full_exposure'] += 1
                                
                        except Exception as record_error:
                            st.warning(f"Error analyzing record {i+1}: {record_error}")
                            continue
                    
                    progress_bar.progress(100)
                    status_text.success("Analysis complete!")
                    
                    # Store results in session state
                    st.session_state.exposure_analysis_results = summary
                    st.session_state.exposure_analysis_running = False
                    
                    # Generate report
                    report = analyzer.generate_report(summary)
                    st.session_state.exposure_report = report
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logger.error(f"Exposure analysis error: {e}", exc_info=True)
                st.session_state.exposure_analysis_running = False
        
        # Display results if available
        if hasattr(st.session_state, 'exposure_analysis_results') and st.session_state.exposure_analysis_results:
            self.display_exposure_results(st.session_state.exposure_analysis_results)
    
    def display_exposure_results(self, summary: Dict[str, Any]):
        """Display exposure analysis results with visualizations"""
        st.header("📊 Exposure Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total = summary['total_records']
        no_exp = summary['no_exposure']
        partial_exp = summary['partial_exposure'] 
        full_exp = summary['full_exposure']
        
        with col1:
            st.metric("Total Records", total)
        
        with col2:
            st.metric("No Exposure", f"{no_exp} ({no_exp/total*100:.1f}%)")
        
        with col3:
            st.metric("Partial Exposure", f"{partial_exp} ({partial_exp/total*100:.1f}%)", 
                     delta=f"{partial_exp}" if partial_exp > 0 else None,
                     delta_color="inverse")
        
        with col4:
            st.metric("Full Exposure", f"{full_exp} ({full_exp/total*100:.1f}%)",
                     delta=f"{full_exp}" if full_exp > 0 else None, 
                     delta_color="inverse")
        
        # Visualization
        if PLOTLY_AVAILABLE and (partial_exp > 0 or full_exp > 0):
            st.subheader("Exposure Distribution")
            
            # Pie chart
            labels = ['No Exposure', 'Partial Exposure', 'Full Exposure']
            values = [no_exp, partial_exp, full_exp]
            colors = ['#00CC96', '#FFA15A', '#EF553B']
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, 
                                       marker=dict(colors=colors))])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title="PII Exposure Levels", width=600, height=400)
            st.plotly_chart(fig, width="stretch")
        
        # Detailed results
        if partial_exp > 0 or full_exp > 0:
            st.subheader("⚠️ Records with PII Exposure")
            
            # Filter exposed records
            exposed_records = [
                record for record in summary['exposure_details'] 
                if record['overall_exposure_level'] != 'NO_EXPOSURE'
            ]
            
            if exposed_records:
                # Display options
                col1, col2 = st.columns([1, 2])
                with col1:
                    max_display = st.slider("Records to display", 1, min(50, len(exposed_records)), 10)
                with col2:
                    show_details = st.checkbox("Show detailed field analysis", True)
                
                # Display records
                for i, record in enumerate(exposed_records[:max_display]):
                    with st.expander(f"Record {record['record_id']} - {record['overall_exposure_level']}", 
                                   expanded=(i == 0)):
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.write(f"**Exposure Level:** {record['overall_exposure_level']}")
                            st.write(f"**Exposed Fields:** {len(record['exposed_fields'])}/{record['total_pii_fields']}")
                        
                        with col2:
                            if 'anonymized_text' in record:
                                st.write("**Anonymized Text Preview:**")
                                st.code(record['anonymized_text'][:200] + "..." if len(record['anonymized_text']) > 200 else record['anonymized_text'])
                        
                        if show_details and record['exposed_fields']:
                            st.write("**Field Details:**")
                            for field_name, field_info in record['exposed_fields'].items():
                                st.write(f"• **{field_name}** ({field_info['exposure_level']})")
                                st.write(f"  - Original: `{field_info['original_value']}`")
                                st.write(f"  - Exposed in text: `{', '.join(field_info['exposed_data'])}`")
                
                if len(exposed_records) > max_display:
                    st.info(f"... and {len(exposed_records) - max_display} more records with exposure")
        
        # Report download
        st.subheader("📄 Export Report")
        col1, col2 = st.columns(2)
        
        with col1:
            if hasattr(st.session_state, 'exposure_report'):
                st.download_button(
                    label="Download Text Report",
                    data=st.session_state.exposure_report,
                    file_name=f"exposure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col2:
            st.download_button(
                label="Download JSON Results", 
                data=json.dumps(summary, indent=2),
                file_name=f"exposure_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def render_status_page(self):
        """Render system status page"""
        st.title("📋 System Status")
        
        # System information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Configuration Files")
            
            # Check analyzer config
            if self.analyzer_config_path.exists():
                st.success("✅ analyzers_config.yaml - OK")
                config = self.load_yaml_file(self.analyzer_config_path)
                if config:
                    analyzers = config.get('analyzers', {})
                    enabled_analyzers = [name for name, conf in analyzers.items() if conf.get('enabled', False)]
                    st.write(f"Active analyzers: {', '.join(enabled_analyzers)}")
            else:
                st.error("❌ analyzers_config.yaml - Missing")
            
            # Check main config
            if self.config_path.exists():
                st.success("✅ config.yaml - OK")
            else:
                st.warning("⚠️ config.yaml - Missing (optional)")
        
        with col2:
            st.subheader("Data Files")
            
            # Check for CSV files in test_datasets folder
            csv_files = list(self.datasets_path.glob("*.csv"))
            if csv_files:
                st.success(f"✅ Found {len(csv_files)} CSV file(s) in test_datasets/")
                for csv_file in csv_files[:5]:  # Show first 5
                    st.write(f"• {csv_file.name}")
                if len(csv_files) > 5:
                    st.write(f"• ... and {len(csv_files) - 5} more")
            else:
                st.warning("⚠️ No CSV files found in test_datasets/")
            
            # Check for results in results folder
            json_files = self.get_json_results_files()
            if json_files:
                st.success(f"✅ Found {len(json_files)} result file(s) in results/")
                latest = json_files[0]
                mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
                st.write(f"Latest: {latest.name}")
                st.write(f"Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.info("ℹ️ No analysis results yet")
        
        # System resources
        st.subheader("System Resources")

        if self._ensure_psutil():
            col3, col4 = st.columns(2)

            with col3:
                cpu_percent = psutil.cpu_percent(interval=1)
                st.metric("CPU Usage", f"{cpu_percent:.1f}%")

                memory = psutil.virtual_memory()
                st.metric("Memory Usage", f"{memory.percent:.1f}%")

            with col4:
                disk = psutil.disk_usage(str(self.base_path))
                disk_percent = (disk.used / disk.total) * 100
                st.metric("Disk Usage", f"{disk_percent:.1f}%")

                st.metric("Available Disk", f"{disk.free / (1024**3):.1f} GB")
        else:
            st.info("psutil is not installed, so host resource metrics are unavailable.")
            st.code(f"{sys.executable} -m pip install psutil", language="bash")
            if st.button("Install psutil", key="install_psutil_button"):
                with st.spinner("Installing psutil…"):
                    if self._install_psutil():
                        _st_rerun()
    
    def render_documentation_page(self):
        """Render documentation page"""
        st.title("📖 Documentation")
        st.markdown(dedent("""
        ## PII Detection Control Center

        Welcome to the hub for managing UK-centric PII experiments. The sidebar keeps navigation lightweight, while the Interactive Search workspace houses the broader toolset.

        ### Interactive Workspace

        - **Ad-hoc search** – Paste text, tune recognizers, and inspect anonymized/tokenized output side by side.
        - **Full-run launcher** – Start threaded analyses, watch progress, and drop JSON artifacts into `results/` without leaving the page.
        - **Inline configuration** – Edit `analyzers_config.yaml`, `config.yaml`, and recognizer YAMLs through the embedded editors.
        - **Results + exposure insights** – Load saved runs, normalize records, and review exposure summaries directly below the search pane.
        - **Simple Presidio pipeline** – Upload CSVs and anonymize them via the lightweight wrapper in-place.

        ### Additional Pages

        - **🖼️ Slides** – Browse supporting reference decks stored under `Slides/`.
        - **📋 System Status** – Validate configs, datasets, and resource usage at a glance.
        - **📖 Documentation** – Reference usage tips, supported entities, and troubleshooting guides.

        ### Configuration Files

        - **analyzers_config.yaml** – Enables/disables analyzers, selects transformer/Ollama options, and defines overlap handling.
        - **config.yaml** – Stores global defaults such as max records, thread count, and logging level.
        - **presidio_recognizers_config.yaml** – Houses pattern-based recognizers that complement code and transformer detectors.

        ### Tips & Troubleshooting

        1. When detections look off, adjust recognizer toggles within Interactive Search before rerunning.
        2. For slow runs, reduce transformer models or enable "Built-in Presidio only" in the configuration column.
        3. Upload fresh CSVs into `test_datasets/` and refresh the page if they do not appear in pickers.
        4. Use **System Status** to diagnose missing dependencies or YAML parse errors.
        5. Exposure reports rely on tokenized fields; if summaries look empty, trigger a new search so the latest payload is cached.
        """))
    
    def run(self):
        """Main application runner"""
        try:
            # Render sidebar
            page = self.render_sidebar()
            
            # Render selected page
            if page == "🔍 Interactive Search":
                self.render_interactive_search_page()
            elif page == "🖼️ Slides":
                self.render_slides_page()
            elif page == "📊 PII Exposure Report":
                self.render_exposure_report_page()
            elif page == "📋 System Status":
                self.render_status_page()
            elif page == "📖 Documentation":
                self.render_documentation_page()
            else:
                self.render_interactive_search_page()
            
        except Exception as e:
            st.error(f"Application error: {e}")
            logger.error(f"Application error: {e}", exc_info=True)

    def render_slides_page(self):
        """Render a simple JPEG gallery with thumbnails and full-view for Slides/ by default."""
        st.title("🖼️ Slides Viewer")
        from pathlib import Path as _Path

        # Defaults to Slides/ under base path
        default_dir = str(self.base_path / "Slides")
        slides_dir = st.text_input("Images folder (JPEG)", value=st.session_state.get("__slides_dir__", default_dir))
        st.session_state["__slides_dir__"] = slides_dir

        p = _Path(slides_dir)
        if not p.exists() or not p.is_dir():
            st.warning(f"Folder not found: {slides_dir}")
            return

        # Find image files (JPEG/PNG, case-insensitive)
        files = []
        for pattern in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            files.extend(sorted(p.glob(pattern)))

        if not files:
            st.info("No JPEG images found in this folder.")
            return

        # Viewing state
        selected = st.session_state.get("__slides_selected__")
        # Maintain list in state for navigation
        st.session_state["__slides_files__"] = [str(f) for f in files]

        if selected:
            # Full image view
            files_list = st.session_state.get("__slides_files__", [])
            try:
                cur_idx = files_list.index(selected)
            except ValueError:
                cur_idx = 0

            st.subheader(f"{_Path(selected).name} ({cur_idx+1}/{len(files_list)})")
            st.image(selected, width="stretch")

            col_a, col_b, col_c, col_d = st.columns([1,1,1,1])
            with col_a:
                if st.button("⬅️ Back to thumbnails", key="slides_back"):
                    st.session_state["__slides_selected__"] = None
                    _st_rerun()
            with col_b:
                if st.button("◀️ Previous", key="slides_prev"):
                    prev_idx = (cur_idx - 1) % len(files_list)
                    st.session_state["__slides_selected__"] = files_list[prev_idx]
                    _st_rerun()
            with col_c:
                if st.button("Next ▶️", key="slides_next"):
                    next_idx = (cur_idx + 1) % len(files_list)
                    st.session_state["__slides_selected__"] = files_list[next_idx]
                    _st_rerun()
            with col_d:
                with open(selected, "rb") as fh:
                    st.download_button("📥 Download", data=fh.read(), file_name=_Path(selected).name)
            return

        # Thumbnails grid
        st.subheader(f"Gallery: {slides_dir}")
        cols_per_row = st.slider("Columns", min_value=2, max_value=8, value=4, help="Number of thumbnails per row")
        cols = st.columns(cols_per_row)

        for idx, fp in enumerate(files):
            col = cols[idx % cols_per_row]
            with col:
                st.image(str(fp), caption=fp.name, width="stretch")
                if st.button("View", key=f"view_{idx}"):
                    st.session_state["__slides_selected__"] = str(fp)
                    _st_rerun()

    def render_simple_presidio_page(self):
        """Render the Simple Presidio Anonymizer page"""
        st.title("🔬 Simple Presidio Anonymizer")
        st.markdown("Upload a CSV file and anonymize PII using standalone Presidio")
        
        if not SIMPLE_PRESIDIO_AVAILABLE:
            st.error("❌ Simple Presidio module not available. Please ensure simple_presidio_anonymizer.py is in the working directory.")
            return
        
        # File upload section
        st.header("📁 Upload CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file containing text data to anonymize"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully: {len(df)} records")
                
                # Show file preview
                with st.expander("📋 File Preview (first 5 rows)", expanded=True):
                    st.dataframe(df.head())
                
                # Configuration section
                st.header("⚙️ Configuration")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Field selection
                    available_fields = list(df.columns)
                    default_field = "source" if "source" in available_fields else available_fields[0]
                    
                    target_field = st.selectbox(
                        "Select field to anonymize",
                        options=available_fields,
                        index=available_fields.index(default_field),
                        help="Choose which column contains the text to anonymize"
                    )
                
                with col2:
                    # Record limit
                    max_records = st.number_input(
                        "Max records to process",
                        min_value=1,
                        max_value=len(df),
                        value=min(100, len(df)),
                        help=f"Limit processing to avoid long waits. File has {len(df)} total records"
                    )
                
                with col3:
                    # Output filename
                    default_output = f"anonymized_{uploaded_file.name.replace('.csv', '')}.csv"
                    output_filename = st.text_input(
                        "Output filename",
                        value=default_output,
                        help="Name for the anonymized output file"
                    )
                
                # Show preview of selected field
                st.subheader(f"📝 Preview of '{target_field}' field")
                
                # Show processing info
                if max_records < len(df):
                    st.info(f"ℹ️ Will process first {max_records} records out of {len(df)} total records")
                else:
                    st.info(f"ℹ️ Will process all {len(df)} records")
                
                sample_texts = df[target_field].dropna().head(3).tolist()
                
                for i, text in enumerate(sample_texts):
                    st.text_area(
                        f"Sample {i+1}",
                        value=str(text),
                        height=60,
                        disabled=True,
                        key=f"sample_text_{i}"
                    )
                
                # Process button
                if st.button("🚀 Run Presidio Anonymization", type="primary", width="stretch"):
                    self.process_simple_presidio(df, target_field, output_filename, uploaded_file.name, max_records)
                    
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {e}")
        
        else:
            # Show example and instructions
            st.info("👆 Upload a CSV file to get started")
            
            st.subheader("📚 Instructions")
            st.markdown("""
            1. **Upload CSV**: Select a CSV file with text data
            2. **Choose Field**: Select which column contains text to anonymize
            3. **Configure Output**: Set the output filename
            4. **Run Analysis**: Click to process with Presidio
            5. **View Results**: See anonymized data and download results
            """)
            
            st.subheader("📄 Expected CSV Format")
            st.markdown("""
            Your CSV should have a header row and at least one column with text data:
            
            ```csv
            id,source,category
            1,"Hi, my name is John Smith and email john@company.com",inquiry
            2,"Call me at 555-123-4567 for more information",contact
            3,"My address is 123 Main St, London SW1A 1AA",address
            ```
            """)
            
            st.subheader("🏷️ Detected PII Types")
            st.markdown("""
            The Simple Presidio Anonymizer automatically detects:
            
            | Entity Type | Examples |
            |-------------|----------|
            | **PERSON** | "John Smith", "Sarah Johnson" |
            | **EMAIL_ADDRESS** | "john@company.com" |
            | **PHONE_NUMBER** | "555-123-4567", "+44 20 7946 0958" |
            | **URL** | "https://example.com" |
            | **IP_ADDRESS** | "192.168.1.1" |
            | **CREDIT_CARD** | "4532-1234-5678-9012" |
            | **LOCATION** | "London", "New York" |
            | **US_SSN** | "123-45-6789" |
            | **DATE_TIME** | "2023-01-01", "January 1st" |
            """)

    def process_simple_presidio(self, df: pd.DataFrame, target_field: str, output_filename: str, original_filename: str, max_records: int):
        """Process the DataFrame with Simple Presidio"""
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Limit the DataFrame if requested
            if max_records < len(df):
                df = df.head(max_records)
                status_text.text(f"📝 Limited to first {max_records} records...")
            else:
                status_text.text("📝 Preparing data...")
            
            progress_bar.progress(10)
            
            # Save DataFrame to temporary file
            import tempfile
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                temp_input_path = temp_file.name
                df.to_csv(temp_input_path, index=False)
            
            # Set up output path
            output_path = os.path.join(os.getcwd(), output_filename)
            
            status_text.text("🔍 Running Presidio analysis...")
            progress_bar.progress(30)
            
            # Process with Simple Presidio
            final_output_path, results = process_csv(
                input_file=temp_input_path,
                output_file=output_path,
                source_field=target_field,
                max_records=max_records  # Pass the record limit
            )
            
            progress_bar.progress(80)
            status_text.text("📊 Processing results...")
            
            # Load results
            result_df = pd.read_csv(final_output_path)
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            # Clean up temp file
            os.unlink(temp_input_path)
            
            # Display results
            st.success(f"🎉 Anonymization complete! Processed {len(results)} records")
            
            # Results summary
            self.display_simple_presidio_results(result_df, results, target_field, original_filename)
            
        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
            logger.error(f"Simple Presidio processing error: {e}", exc_info=True)
        
        finally:
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()

    def display_simple_presidio_results(self, result_df: pd.DataFrame, results: List[Dict], target_field: str, original_filename: str):
        """Display the results from Simple Presidio processing"""
        
        # Summary metrics
        st.header("📊 Results Summary")
        
        total_records = len(results)
        records_with_pii = sum(1 for r in results if r['entities_found'] > 0)
        total_entities = sum(r['entities_found'] for r in results)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", total_records)
        with col2:
            st.metric("Records with PII", records_with_pii)
        with col3:
            st.metric("Total Entities", total_entities)
        with col4:
            pii_percentage = (records_with_pii / total_records * 100) if total_records > 0 else 0
            st.metric("PII Detection Rate", f"{pii_percentage:.1f}%")
        
        # Entity type distribution
        entity_types = {}
        for result in results:
            for entity in result['entities']:
                entity_type = entity['entity_type']
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        if entity_types:
            st.subheader("🏷️ Entity Types Found")
            entity_df = pd.DataFrame([
                {'Entity Type': etype, 'Count': count} 
                for etype, count in sorted(entity_types.items())
            ])
            st.dataframe(entity_df, width="stretch", hide_index=True)
        
        # Results tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Sample Results", "📋 Full Results", "💾 Download"])
        
        with tab1:
            st.subheader("Sample Anonymized Results")
            
            # Show first few records with PII
            sample_records = [r for r in results if r['entities_found'] > 0][:5]
            
            for i, record in enumerate(sample_records):
                with st.expander(f"Record {record['record_id']} - {record['entities_found']} entities found"):
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Text:**")
                        st.text_area(
                            "Original",
                            value=record['original_text'],
                            height=100,
                            disabled=True,
                            key=f"orig_{i}"
                        )
                    
                    with col2:
                        st.markdown("**Anonymized Text:**")
                        st.text_area(
                            "Anonymized", 
                            value=record['anonymized_text'],
                            height=100,
                            disabled=True,
                            key=f"anon_{i}"
                        )
                    
                    # Show detected entities
                    if record['entities']:
                        st.markdown("**Detected Entities:**")
                        entity_display = []
                        for entity in record['entities']:
                            entity_display.append({
                                'Type': entity['entity_type'],
                                'Text': entity['text'],
                                'Confidence': f"{entity['confidence']:.2f}",
                                'Position': f"{entity['start']}-{entity['end']}"
                            })
                        
                        st.dataframe(
                            pd.DataFrame(entity_display), 
                            width="stretch", 
                            hide_index=True
                        )
        
        with tab2:
            st.subheader("Complete Results Dataset")
            
            # Show key columns
            display_columns = [col for col in result_df.columns if col in [
                'id', target_field, f'{target_field}_anonymized', 'entities_found', 'entity_types'
            ]]
            
            if display_columns:
                st.dataframe(
                    result_df[display_columns],
                    width="stretch",
                    height=400
                )
            else:
                st.dataframe(result_df, width="stretch", height=400)
            
            # Search functionality
            st.subheader("🔍 Search Results")
            search_term = st.text_input("Search in results", placeholder="Enter search term...")
            
            if search_term:
                # Search in original and anonymized text
                mask = (
                    result_df[target_field].astype(str).str.contains(search_term, case=False, na=False) |
                    result_df[f'{target_field}_anonymized'].astype(str).str.contains(search_term, case=False, na=False)
                )
                
                filtered_results = result_df[mask]
                st.write(f"Found {len(filtered_results)} matching records:")
                
                if len(filtered_results) > 0:
                    st.dataframe(filtered_results, width="stretch")
                else:
                    st.info("No matching records found")
        
        with tab3:
            st.subheader("📥 Download Results")
            
            # Prepare CSV data
            csv_data = result_df.to_csv(index=False)
            
            # Download button
            st.download_button(
                label="📥 Download Anonymized CSV",
                data=csv_data,
                file_name=f"anonymized_{original_filename}",
                mime="text/csv",
                type="primary",
                width="stretch"
            )
            
            st.markdown("**File includes:**")
            st.markdown(f"""
            - All original columns from your CSV
            - `{target_field}_anonymized` - The anonymized version
            - `entities_found` - Number of PII entities detected
            - `entity_types` - Types of entities found
            - `entity_details` - Detailed entity information
            """)

def main():
    """Main entry point"""
    control_center = PIIControlCenter()
    control_center.run()

if __name__ == "__main__":
    main()
