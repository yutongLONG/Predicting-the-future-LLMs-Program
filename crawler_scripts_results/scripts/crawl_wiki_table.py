"""
Wikipedia Table Extractor - Extract fundraising tables from Wikipedia
Extract all 2024 campaign financing tables and save as CSV
Based on crawl_web.py structure for consistency
"""
import asyncio
import sys
import json
import csv
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup

# Windows needs to set the event loop policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig


def clean_cell_text(cell) -> str:
    """
    Clean cell text: remove references, links, keep only text content
    """
    if cell is None:
        return ""
    
    # Get text content, removing all HTML tags
    text = cell.get_text(separator=' ', strip=True)
    
    # Remove citation marks [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&mdash;', '—')
    text = text.replace('&ndash;', '–')
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_caption_text_full(caption) -> str:
    """
    Extract full text from caption, preserving all text nodes in order
    One character should not be missed!
    """
    if caption is None:
        return ""
    
    # Method 1: Use stripped_strings to get all text nodes
    # This preserves the order and gets all visible text
    text_parts = list(caption.stripped_strings)
    
    if text_parts:
        # Join all text parts with space
        full_text = ' '.join(text_parts)
    else:
        # Fallback: Use get_text if stripped_strings is empty
        full_text = caption.get_text(separator=' ', strip=True)
    
    # Only remove citation marks [1], [2], etc.
    # Keep everything else including special characters
    full_text = re.sub(r'\[\d+\]', '', full_text)
    
    # Handle HTML entities (but preserve their meaning)
    full_text = full_text.replace('&nbsp;', ' ')
    full_text = full_text.replace('&amp;', '&')
    full_text = full_text.replace('&mdash;', '—')
    full_text = full_text.replace('&ndash;', '–')
    full_text = full_text.replace('&quot;', '"')
    full_text = full_text.replace('&apos;', "'")
    full_text = full_text.replace('&lt;', '<')
    full_text = full_text.replace('&gt;', '>')
    
    # Clean up multiple spaces but preserve single spaces
    full_text = re.sub(r'\s+', ' ', full_text)
    
    return full_text.strip()



def is_relevant_table(table) -> bool:
    """
    Check if table is relevant to 2024 campaign financing
    """
    # First check: loose 2024 filtering
    headings = []
    current = table.find_previous(['h2', 'h3'])
    while current:
        heading_text = clean_cell_text(current).lower()
        headings.append(heading_text)
        current = current.find_previous(['h2', 'h3'])
    
    all_heading_text = ' '.join(headings)
    # Loose filter: skip if clearly 2023 and no 2024
    if '2023' in all_heading_text and '2024' not in all_heading_text:
        return False
    
    # Original checks
    # Check table caption
    caption = table.find('caption')
    if caption:
        caption_text = caption.get_text().lower()
        if '2024' in caption_text and any(keyword in caption_text for keyword in 
            ['fundraising', 'campaign', 'financing', 'raised', 'spent', 'cash']):
            return True
    
    # Check table class
    table_class = table.get('class', [])
    if 'wikitable' in ' '.join(table_class):
        # Check first few rows for candidate names or financial data
        rows = table.find_all('tr', limit=5)
        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_text = ' '.join([cell.get_text().lower() for cell in cells])
            if any(name in row_text for name in ['harris', 'trump', 'kennedy', 'west', 'biden']):
                return True
            if '$' in row_text or 'raised' in row_text or 'spent' in row_text:
                return True
    
    return False


def extract_table_title(table, context: str = "") -> str:
    """
    Extract table title from caption or context
    Improved version that preserves all text, one character should not be missed!
    """
    # Priority 1: Extract from caption (most accurate)
    caption = table.find('caption')
    if caption:
        title = extract_caption_text_full(caption)
        if title and title.strip():
            return title
    
    # Priority 2: Try to find preceding heading
    prev = table.find_previous(['h2', 'h3', 'h4'])
    if prev:
        title = extract_caption_text_full(prev)  # Reuse the same function
        if title and title.strip():
            # Only add " - Table" if it doesn't already contain "Table"
            if '2024' in title and 'table' not in title.lower():
                return f"{title} - Table"
            return title
    
    # Priority 3: Use context if provided
    if context:
        return f"Table - {context}"
    
    # Priority 4: Default fallback
    return "Campaign Financing Table"

def extract_time_hierarchy(table) -> Optional[Dict]:
    """
    Extract time hierarchy from headings (h2 and h3 only)
    Returns None if table is from 2023, otherwise returns time info
    """
    # Find all headings before the table
    headings = []
    current = table.find_previous(['h2', 'h3'])
    
    # Collect all h2 and h3 headings before the table
    while current:
        heading_text = clean_cell_text(current)
        headings.append((current.name, heading_text))
        current = current.find_previous(['h2', 'h3'])
    
    # Reverse to get chronological order (h2 first, then h3)
    headings.reverse()
    
    # Check for 2023 (loose filtering)
    all_text = ' '.join([text for _, text in headings]).lower()
    if '2023' in all_text and '2024' not in all_text:
        return None  # Skip 2023 tables
    
    # Extract year and period
    year = "2024"
    period = None
    context_chain = []
    
    for tag, text in headings:
        if tag == 'h2' and '2024' in text:
            year = "2024"
            context_chain.append(text)
        elif tag == 'h3':
            # Extract period (month or Post-General, Pre-General, etc.)
            period = text
            context_chain.append(text)
            break  # Only take the first h3
    
    # Build full path
    if period:
        full_path = f"{year} > {period}"
    else:
        full_path = year
    
    return {
        "year": year,
        "period": period,
        "full_path": full_path,
        "context_chain": context_chain
    }

def extract_headers(table) -> List[str]:
    """
    Extract table headers, handling multi-level headers and merged cells
    """
    headers = []
    thead = table.find('thead')
    
    if thead:
        # Process header rows
        header_rows = thead.find_all('tr')
        if header_rows:
            # Handle multi-level headers
            if len(header_rows) > 1:
                # Combine headers from multiple rows
                header_cells = []
                for row in header_rows:
                    cells = row.find_all(['th', 'td'])
                    row_headers = []
                    for cell in cells:
                        text = clean_cell_text(cell)
                        rowspan = int(cell.get('rowspan', 1))
                        colspan = int(cell.get('colspan', 1))
                        
                        if colspan > 1:
                            # Split into multiple columns
                            for _ in range(colspan):
                                row_headers.append(text)
                        else:
                            row_headers.append(text)
                    
                    header_cells.append(row_headers)
                
                # Combine multi-level headers
                max_cols = max(len(row) for row in header_cells)
                combined_headers = []
                for i in range(max_cols):
                    parts = []
                    for row in header_cells:
                        if i < len(row) and row[i]:
                            parts.append(row[i])
                    combined_headers.append(' - '.join(filter(None, parts)) if parts else f"Column_{i+1}")
                
                headers = combined_headers
            else:
                # Single row header
                cells = header_rows[0].find_all(['th', 'td'])
                for cell in cells:
                    text = clean_cell_text(cell)
                    colspan = int(cell.get('colspan', 1))
                    if colspan > 1:
                        for i in range(colspan):
                            headers.append(f"{text}_{i+1}" if i > 0 else text)
                    else:
                        headers.append(text)
    else:
        # No thead, use first row
        first_row = table.find('tr')
        if first_row:
            cells = first_row.find_all(['th', 'td'])
            for cell in cells:
                text = clean_cell_text(cell)
                headers.append(text)
    
    # Clean and validate headers
    cleaned_headers = []
    for i, header in enumerate(headers):
        if not header or header.strip() == '':
            header = f"Column_{i+1}"
        cleaned_headers.append(header)
    
    return cleaned_headers


def extract_table_rows(table, num_headers: int) -> List[List[str]]:
    """
    Extract data rows from table, handling merged cells
    """
    rows = []
    tbody = table.find('tbody')
    table_body = tbody if tbody else table
    
    data_rows = table_body.find_all('tr')
    
    # Track rowspan values for each column
    rowspan_tracker = {}
    
    for row_idx, row in enumerate(data_rows):
        cells = row.find_all(['td', 'th'])
        
        # Skip if this looks like a header row (all th tags)
        if all(cell.name == 'th' for cell in cells) and row_idx == 0:
            continue
        
        row_data = []
        col_idx = 0
        
        for cell in cells:
            # Handle rowspan - fill in previous values
            while col_idx in rowspan_tracker:
                row_data.append(rowspan_tracker[col_idx])
                col_idx += 1
            
            # Handle colspan
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            
            text = clean_cell_text(cell)
            
            # Add current cell
            row_data.append(text)
            col_idx += 1
            
            # Handle colspan - duplicate value
            for _ in range(colspan - 1):
                row_data.append(text)
                col_idx += 1
            
            # Handle rowspan - store value for future rows
            if rowspan > 1:
                for i in range(1, rowspan):
                    future_row = row_idx + i
                    if future_row not in rowspan_tracker:
                        rowspan_tracker[future_row] = {}
                    rowspan_tracker[future_row][col_idx - 1] = text
        
        # Fill remaining columns from rowspan tracker
        while col_idx < num_headers:
            if col_idx in rowspan_tracker:
                row_data.append(rowspan_tracker[col_idx])
            else:
                row_data.append("")
            col_idx += 1
        
        # Only add row if it has data
        if any(cell.strip() for cell in row_data):
            # Ensure row has correct number of columns
            while len(row_data) < num_headers:
                row_data.append("")
            rows.append(row_data[:num_headers])
    
    return rows



def parse_tables_from_html(html: str, base_url: str = "") -> List[Dict]:
    """
    Parse all relevant tables from HTML
    """
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    
    extracted_tables = []
    
    for idx, table in enumerate(tables):
        if not is_relevant_table(table):
            continue
        
        try:
            # Extract time hierarchy
            time_hierarchy = extract_time_hierarchy(table)
            if time_hierarchy is None:
                continue  # Skip 2023 tables
            
            # Extract table title
            title = extract_table_title(table)
            
            # Extract headers
            headers = extract_headers(table)
            if not headers:
                continue
            
            # Extract rows
            rows = extract_table_rows(table, len(headers))
            if not rows:
                continue
            
            # Get context (preceding heading)
            context = ""
            prev_heading = table.find_previous(['h2', 'h3'])
            if prev_heading:
                context = clean_cell_text(prev_heading)
            
            table_data = {
                "table_index": idx + 1,
                "table_title": title,
                "context": context,
                "time_hierarchy": time_hierarchy,  # Add time hierarchy
                "headers": headers,
                "rows": rows,
                "row_count": len(rows),
                "column_count": len(headers)
            }
            
            extracted_tables.append(table_data)
            
        except Exception as e:
            print(f"  ⚠️  Error extracting table {idx + 1}: {e}")
            continue
    
    return extracted_tables


def save_table_to_csv(table_data: Dict, output_path: Path, prefix: str = "table") -> str:
    """
    Save table data to CSV file
    """
    # Generate filename
    table_idx = table_data['table_index']
    title = table_data['table_title']
    
    # Clean title for filename
    safe_title = re.sub(r'[^\w\s-]', '', title)
    safe_title = re.sub(r'[-\s]+', '_', safe_title)
    safe_title = safe_title[:50]  # Limit length
    
    filename = f"{prefix}_{table_idx:02d}_{safe_title}.csv"
    csv_file = output_path / filename
    
    # Write CSV
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:  # utf-8-sig for Excel compatibility
        writer = csv.writer(f)
        
        # Write headers
        writer.writerow(table_data['headers'])
        
        # Write rows
        for row in table_data['rows']:
            writer.writerow(row)
    
    return str(csv_file)

def merge_tables_to_csv(tables: List[Dict], output_path: Path, prefix: str = "merged") -> str:
    """
    Merge all tables into a single CSV with time columns
    """
    if not tables:
        raise ValueError("No tables to merge")
    
    # Determine all unique headers across tables
    all_headers = set()
    for table in tables:
        all_headers.update(table['headers'])
    
    # Standardize header order (put time columns first, then others)
    time_headers = ['Year', 'Period', 'Time_Path']
    other_headers = sorted([h for h in all_headers if h not in time_headers])
    merged_headers = time_headers + other_headers
    
    # Merge all rows
    merged_rows = []
    for table in tables:
        time_info = table.get('time_hierarchy', {})
        year = time_info.get('year', '2024')
        period = time_info.get('period', '')
        time_path = time_info.get('full_path', '2024')
        table_title = table.get('table_title', '')  # 获取标题
        
        for row in table['rows']:
            # Create a dictionary for this row
            row_dict = dict(zip(table['headers'], row))
            
            # Add time columns and title
            merged_row = [
                year,
                period,
                time_path,
                table_title  # 添加标题列
            ]
            
            # Add other columns
            for header in other_headers:
                merged_row.append(row_dict.get(header, ''))
            
            merged_rows.append(merged_row)
    
    # Sort by period (Post-General > Pre-General > months)
    def period_sort_key(row):
        period = row[1]  # Period is second column
        if not period:
            return (3, '')
        period_lower = period.lower()
        if 'post-general' in period_lower:
            return (0, period)
        elif 'pre-general' in period_lower:
            return (1, period)
        else:
            # Month order: September (9) to January (1)
            month_order = {
                'september': 9, 'august': 8, 'july': 7, 'june': 6,
                'may': 5, 'april': 4, 'march': 3, 'february': 2, 'january': 1
            }
            for month, order in month_order.items():
                if month in period_lower:
                    return (2, -order)  # Negative for descending order
            return (2, period)
    
    merged_rows.sort(key=period_sort_key)
    
    # Write merged CSV
    filename = f"{prefix}_merged.csv"
    csv_file = output_path / filename
    
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(merged_headers)
        writer.writerows(merged_rows)
    
    return str(csv_file)


async def extract_wiki_tables(
    url: str,
    output_dir: str = "wiki_results",
    output_prefix: str = "fundraising_2024",
    headless: bool = True,
    verbose: bool = True,
    save_html: bool = False
):
    """
    Extract all 2024 campaign financing tables from Wikipedia page
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting to extract campaign financing tables")
    print(f"📄 Page: {url}")
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"📝 File prefix: {output_prefix}")
    print("-" * 60)
    
    # Configure browser
    browser_config = BrowserConfig(
        headless=headless,
        verbose=verbose
    )
    
    # Configure crawling parameters
    crawler_config = CrawlerRunConfig(
        wait_for=None,
        css_selector=None,
        delay_before_return_html=3.0,  # Wait for tables to load
        word_count_threshold=1,
        excluded_tags=['nav', 'header', 'footer', 'script', 'style'],
        cache_mode="bypass"
    )
    
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            print("📋 Step 1: Loading page...")
            result = await crawler.arun(
                url=url,
                config=crawler_config
            )
            
            if result.success:
                print("✅ Page load successful")
                
                # Save HTML if requested
                if save_html:
                    html_file = output_path / f"{output_prefix}.html"
                    with open(html_file, "w", encoding="utf-8") as f:
                        f.write(result.html)
                    print(f"✅ HTML saved: {html_file.name}")
                
                # Parse tables
                print("📋 Step 2: Parsing tables...")
                html_content = result.html
                tables = parse_tables_from_html(html_content, url)
                
                if not tables:
                    print("❌ No relevant tables found")
                    return {
                        "success": False,
                        "error": "No relevant tables found"
                    }
                
                print(f"✅ Found {len(tables)} relevant table(s)")
                
                
                # Merge and save all tables to a single CSV
                print("📋 Step 4: Merging tables and saving to CSV...")
                merged_csv = merge_tables_to_csv(tables, output_path, output_prefix)
                print(f"  ✅ Merged CSV saved: {Path(merged_csv).name}")
                print(f"     Total rows: {sum(t['row_count'] for t in tables)}")
                
                # Keep individual CSV files list for metadata (optional, can be empty)
                csv_files = [merged_csv]


                # Save metadata
                metadata = {
                    "url": url,
                    "extracted_at": datetime.now().isoformat(),
                    "table_count": len(tables),
                    "merged_csv": Path(merged_csv).name,
                    "total_rows": sum(t['row_count'] for t in tables),
                    "tables": [
                        {
                            "index": t['table_index'],
                            "title": t['table_title'],
                            "context": t['context'],
                            "time_hierarchy": t.get('time_hierarchy', {}),
                            "rows": t['row_count'],
                            "columns": t['column_count']
                        }
                        for t in tables
                    ]
                }
                
                metadata_file = output_path / f"{output_prefix}_metadata.json"
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                print(f"✅ Metadata saved: {metadata_file.name}")

                print("-" * 60)
                print(f"📊 Summary:")
                print(f"   - Tables extracted: {len(tables)}")
                print(f"   - Merged CSV: {Path(merged_csv).name}")
                print(f"   - Total rows: {sum(t['row_count'] for t in tables)}")
                print("-" * 60)
                
                return {
                    "success": True,
                    "tables": tables,
                    "files": {
                        "merged_csv": merged_csv,
                        "metadata": str(metadata_file),
                        "html": str(html_file) if save_html else None
                    },
                    "metadata": metadata
                }

            else:
                print(f"❌ Page load failed: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
                return {
                    "success": False,
                    "error": result.error_message if hasattr(result, 'error_message') else "Unknown error"
                }
    
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def main():
    """Command line entry"""
    parser = argparse.ArgumentParser(
        description="Extract campaign financing tables from Wikipedia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python crawl_wiki_table.py https://en.wikipedia.org/wiki/Fundraising_in_the_2024_United_States_presidential_election
  python crawl_wiki_table.py <url> -o ./output -p fundraising_tables --save-html
        """
    )
    
    parser.add_argument("url", help="Wikipedia URL to extract tables from")
    parser.add_argument("-o", "--output-dir", default="wiki_results",
                       help="Output directory (default: wiki_results)")
    parser.add_argument("-p", "--prefix", default="fundraising_2024",
                       help="Output file prefix (default: fundraising_2024)")
    parser.add_argument("--show-browser", action="store_true",
                       help="Show browser window")
    parser.add_argument("--save-html", action="store_true",
                       help="Save original HTML")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Quiet mode")
    
    args = parser.parse_args()
    
    result = asyncio.run(extract_wiki_tables(
        url=args.url,
        output_dir=args.output_dir,
        output_prefix=args.prefix,
        headless=not args.show_browser,
        verbose=not args.quiet,
        save_html=args.save_html
    ))
    
    if result["success"]:
        print("\n✅ Extraction completed!")
        sys.exit(0)
    else:
        print(f"\n❌ Extraction failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()