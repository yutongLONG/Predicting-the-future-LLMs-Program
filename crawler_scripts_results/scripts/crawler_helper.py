"""
Jupyter Notebook Web Scraping Assistant Module
Provides convenient functions for invoking web scraping scripts within notebooks
"""
import subprocess
import sys
import json
from pathlib import Path


def crawl(
    url: str,
    output_dir: str = "crawl_results",
    prefix: str = None,
    wait_for: str = None,
    css_selector: str = None,
    wait_time: float = 0,
    show_browser: bool = False,
    save_html: bool = False,
    return_content: bool = True
):
    """
    Calling Crawler Scripts in Jupyter
    
    Parameters:
        url: Target URL
        output_dir: Output directory
        prefix: File prefix
        wait_for: Element to wait for (CSS selector)
        css_selector: Extract specific element
        wait_time: wait duration (seconds)
        show_browser: whether to display the browser
        save_html: whether to save HTML
        return_content: whether to return content
    
    Returns:
        dict: containing success, content, metadata, etc.
    """
    # Construct command
    cmd = [sys.executable, "crawl_web.py", url]
    cmd.extend(["-o", output_dir])
    
    if prefix:
        cmd.extend(["-p", prefix])
    if wait_for:
        cmd.extend(["--wait-for", wait_for])
    if css_selector:
        cmd.extend(["--css-selector", css_selector])
    if wait_time > 0:
        cmd.extend(["--wait-time", str(wait_time)])
    if show_browser:
        cmd.append("--no-headless")
    if save_html:
        cmd.append("--save-html")
    
    # Run the script
    print(f" Executing command: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    print(result.stdout)
    
    if result.stderr:
        print(" Error information:")
        print(result.stderr)
    
    # Read the results
    if return_content and result.returncode == 0:
        try:
            # Find the generated files
            output_path = Path(output_dir)
            md_files = list(output_path.glob("*.md"))
            
            if md_files:
                latest_md = max(md_files, key=lambda p: p.stat().st_mtime)
                
                # Read the Markdown
                with open(latest_md, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Read the metadata
                metadata_file = latest_md.with_name(
                    latest_md.stem + "_metadata.json"
                )
                if metadata_file.exists():
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                else:
                    metadata = {}
                
                return {
                    "success": True,
                    "content": content,
                    "metadata": metadata,
                    "files": {
                        "markdown": str(latest_md),
                        "metadata": str(metadata_file) if metadata_file.exists() else None
                    }
                }
        except Exception as e:
            print(f" Error reading results: {e}")
    
    return {
        "success": result.returncode == 0,
        "returncode": result.returncode
    }


def batch_crawl(urls: list, output_dir: str = "crawl_results", **kwargs):
    """
    Batch crawl multiple URLs
    
    Parameters:
        urls: List of URLs
        output_dir: Output directory
        **kwargs: Other parameters passed to crawl()
    
    Returns:
        list: Crawling results for each URL
    """
    results = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n{'='*60}")
        print(f" [{i}/{len(urls)}] Crawling: {url}")
        print(f"{'='*60}\n")
        
        result = crawl(url, output_dir=output_dir, **kwargs)
        results.append({
            "url": url,
            "result": result
        })
    
    print(f"\n{'='*60}")
    print(f" Batch crawling completed! Success: {sum(1 for r in results if r['result']['success'])}/{len(urls)}")
    print(f"{'='*60}")
    
    return results



def crawl_wiki_section(
    url: str,
    section_id: str,
    output_dir: str = "wiki_results",
    prefix: str = None,
    show_browser: bool = False,
    save_html: bool = False,
    return_data: bool = True
):
  
    # Construct command
    cmd = [sys.executable, "crawl_wiki_section.py", url, section_id]
    cmd.extend(["-o", output_dir])
    
    if prefix:
        cmd.extend(["-p", prefix])
    if show_browser:
        cmd.append("--no-headless")
    if save_html:
        cmd.append("--save-html")
    
    # Run the script
    print(f" Executing command: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    print(result.stdout)
    
    if result.stderr:
        print(" Error information:")
        print(result.stderr)
    

        # Read the results
    if return_data and result.returncode == 0:
        try:
            # Find the generated JSON files
            # First try the path relative to the script directory
            script_dir = Path(__file__).parent
            output_path = script_dir / output_dir
            
            # If it doesn't exist, try the absolute path
            if not output_path.exists():
                output_path = Path(output_dir)
                if not output_path.is_absolute():
                    # If it's a relative path, try from the current working directory
                    output_path = Path.cwd() / output_dir
            
            if not output_path.exists():
                print(f" Output directory does not exist: {output_path}")
                print(f"   Trying to find JSON files...")
                # Try to find in the script directory
                output_path = script_dir
            
            json_files = list(output_path.glob("*.json"))
            
            # If still not found, try recursive search
            if not json_files:
                json_files = list(output_path.rglob("*.json"))
            
            if json_files:
                latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
                
                with open(latest_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                return {
                    "success": True,
                    "data": data,
                    "files": {
                        "json": str(latest_json)
                    }
                }
            else:
                print(f" No JSON files found in {output_path}")
                if output_path.exists():
                    print(f"   Directory content: {[f.name for f in output_path.iterdir()][:10]}")
        except Exception as e:
            print(f" Error reading results: {e}")
            import traceback
            traceback.print_exc()

    return {
        "success": result.returncode == 0,
        "returncode": result.returncode
    }

def clean_wiki_data(
    json_file: str,
    output_format: str = "json",
    llm_ready: bool = True
):
    """
    Clean and format Wikipedia JSON data
    
    Parameters:
        json_file: JSON file path
        output_format: Output format ("json", "text", "summary")
        llm_ready: Whether to generate a format suitable for LLM
    
    Returns:
        dict: containing the path of the cleaned data
    """
    import subprocess
    
    cmd = [sys.executable, "clean_wiki_data.py", json_file]
    
    if output_format != "json":
        cmd.extend(["--format", output_format])
    if llm_ready:
        cmd.append("--llm-ready")
    
    print(f" Executing cleanup command: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    print(result.stdout)
    
    if result.stderr:
        print(" Error information:")
        print(result.stderr)
    
    if result.returncode == 0:
        # Return the path of the cleaned files
        json_path = Path(json_file)
        cleaned_json = json_path.parent / f"{json_path.stem}_cleaned.json"
        cleaned_txt = json_path.parent / f"{json_path.stem}_cleaned.txt" if llm_ready else None
        
        return {
            "success": True,
            "files": {
                "cleaned_json": str(cleaned_json) if cleaned_json.exists() else None,
                "cleaned_text": str(cleaned_txt) if cleaned_txt and cleaned_txt.exists() else None
            }
        }
    
    return {
        "success": False,
        "returncode": result.returncode
    }

def crawl_timeline_2024(
    url: str,  # Remove default value, make it a required parameter
    output_dir: str = "wiki_results",
    prefix: str = "timeline_2024",
    show_browser: bool = False,
    save_html: bool = False,
    return_data: bool = True
):
    """
    Extract the 2024 timeline (pure text, no links)
    
    Parameters:
        url: Wikipedia timeline page URL (required parameter)
        output_dir: Output directory
        prefix: Output file prefix
        show_browser: Whether to display the browser window
        save_html: Whether to save HTML
        return_data: Whether to return data
    
    Returns:
        dict: containing the extraction results
    """
    import subprocess
    
    script_path = Path(__file__).parent / "crawl_wiki_timeline.py"
    
    cmd = [sys.executable, str(script_path), url]  # url is passed as a positional argument
    cmd.extend(["-o", output_dir])
    cmd.extend(["-p", prefix])
    
    if show_browser:
        cmd.append("--show-browser")
    if save_html:
        cmd.append("--save-html")
    
    print(f" Executing command: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    print(result.stdout)
    
    if result.stderr:
        print(" Error information:")
        print(result.stderr)
    
    # Read the results
    if return_data and result.returncode == 0:
        try:
            script_dir = Path(__file__).parent
            output_path = script_dir / output_dir
            
            if not output_path.exists():
                output_path = Path(output_dir)
            
            json_files = list(output_path.glob(f"{prefix}*.json"))
            
            if json_files:
                latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
                
                with open(latest_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                return {
                    "success": True,
                    "data": data,
                    "files": {
                        "json": str(latest_json),
                        "txt": str(latest_json).replace('.json', '.txt')
                    }
                }
            else:
                print(f" No JSON files found in {output_path}")
        except Exception as e:
            print(f" Error reading results: {e}")
            import traceback
            traceback.print_exc()
    
    return {
        "success": result.returncode == 0,
        "returncode": result.returncode
    }

def crawl_wiki_table(
    url: str,
    output_dir: str = "wiki_results",
    prefix: str = "fundraising_2024",
    show_browser: bool = False,
    save_html: bool = False,
    return_data: bool = True
):
    """
    Extract campaign financing tables from Wikipedia page
    
    Parameters:
        url: Wikipedia page URL
        output_dir: Output directory
        prefix: Output file prefix
        show_browser: Whether to display the browser window
        save_html: Whether to save HTML
        return_data: Whether to return extracted data
    
    Returns:
        dict: containing extraction results
    """
    import subprocess
    
    script_path = Path(__file__).parent / "crawl_wiki_table.py"
    
    cmd = [sys.executable, str(script_path), url]
    cmd.extend(["-o", output_dir])
    cmd.extend(["-p", prefix])
    
    if show_browser:
        cmd.append("--show-browser")
    if save_html:
        cmd.append("--save-html")
    
    print(f" Executing command: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    print(result.stdout)
    
    if result.stderr:
        print(" Error information:")
        print(result.stderr)
    
    # Read the results
    if return_data and result.returncode == 0:
        try:
            script_dir = Path(__file__).parent
            output_path = script_dir / output_dir
            
            if not output_path.exists():
                output_path = Path(output_dir)
            
            # Find CSV files
            csv_files = list(output_path.glob(f"{prefix}_*.csv"))
            
            # Find metadata file
            metadata_file = output_path / f"{prefix}_metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            return {
                "success": True,
                "tables": metadata.get("tables", []),
                "files": {
                    "csv_files": [str(f) for f in csv_files],
                    "metadata": str(metadata_file) if metadata_file.exists() else None
                },
                "metadata": metadata
            }
        except Exception as e:
            print(f" Error reading results: {e}")
            import traceback
            traceback.print_exc()
    
    return {
        "success": result.returncode == 0,
        "returncode": result.returncode
    }