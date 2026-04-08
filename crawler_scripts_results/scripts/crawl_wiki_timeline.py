"""
Wikipedia Timeline Extractor - Extract 2024 timeline only (pure text, no links)
Extract up to November 3, 2024 (excluding November 4 and later)
Based on crawl_web.py structure for stability
"""
import asyncio
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

# Windows needs to set the event loop policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig


async def extract_timeline_2024(
    url: str,
    output_dir: str = "wiki_results",
    output_prefix: str = "timeline_2024",
    headless: bool = True,
    verbose: bool = True,
    save_html: bool = False
):
    """
    Extract 2024 timeline section from Wikipedia timeline page (pure text only)
    Extract up to November 3, 2024 (excluding November 4 and later)
    Reference: crawl_web.py structure for stability
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting to extract 2024 Timeline (up to November 3, 2024)")
    print(f"📄 Page: {url}")
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"📝 File prefix: {output_prefix}")
    print("-" * 60)
    
    # Configure browser - 参考 crawl_web.py
    browser_config = BrowserConfig(
        headless=headless,
        verbose=verbose
    )
    
    # Configure crawling parameters - 参考 crawl_web.py
    crawler_config = CrawlerRunConfig(
        wait_for=None,
        css_selector=None,
        delay_before_return_html=3.0,
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
                
                
                if save_html:
                    html_file = output_path / f"{output_prefix}.html"
                    with open(html_file, "w", encoding="utf-8") as f:
                        f.write(result.html)
                    print(f"✅ HTML saved: {html_file.name}")
                
                
                html_content = result.html
                markdown_content = result.markdown if hasattr(result, 'markdown') else None
                
                # Parse timeline content (pure text only, up to Nov 3)
                print("📋 Step 2: Parsing 2024 timeline content (up to November 3)...")
                timeline_data = parse_timeline_2024(
                    html=html_content,
                    markdown=markdown_content,
                    base_url=url
                )
                
                if not timeline_data:
                    print("❌ Failed to extract timeline content")
                    return {"success": False, "error": "Failed to extract timeline"}
                
                print("✅ Successfully extracted timeline content")
                print(f"   - Title: {timeline_data['section_title']}")
                print(f"   - Sub-sections: {timeline_data['metadata']['subsection_count']}")
                print(f"   - Word count: {timeline_data['metadata']['word_count']:,}")
                
                # Save JSON
                json_file = output_path / f"{output_prefix}.json"
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(timeline_data, f, ensure_ascii=False, indent=2)
                print(f"✅ JSON saved: {json_file.name}")
                
                # Save plain text (for LLM)
                txt_file = output_path / f"{output_prefix}.txt"
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(format_timeline_as_text(timeline_data))
                print(f"✅ Plain text saved: {txt_file.name}")
                
                print("-" * 60)
                print("✅ Extraction completed!")
                print(f"📊 Statistics:")
                print(f"   - Word count: {timeline_data['metadata']['word_count']:,}")
                print(f"   - Paragraph count: {timeline_data['metadata']['paragraph_count']}")
                print(f"   - Sub-section count: {timeline_data['metadata']['subsection_count']}")
                print("=" * 60)
                
                return {
                    "success": True,
                    "data": timeline_data,
                    "files": {
                        "json": str(json_file),
                        "txt": str(txt_file),
                        "html": str(html_file) if save_html else None
                    }
                }
            else:
                error_msg = result.error_message if hasattr(result, 'error_message') else 'Unknown error'
                print(f"❌ Page load failed: {error_msg}")
                return {"success": False, "error": "Page load failed"}
    
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def clean_text(text: str) -> str:
    """
    Clean text: remove links, citations, images (pure text only)
    Enhanced version with comprehensive citation removal
    """
    if not text:
        return ""
    
    # Remove Markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove image links
    text = re.sub(r'!?\[.*?\]\(.*?\)', '', text)
    
    # Remove citation marks - 多种格式
    text = re.sub(r'\[\[(\d+)\]\]', '', text)  # [[212]]
    text = re.sub(r'\[(\d+)\]', '', text)  # [212]
    text = re.sub(r'\[\[(\d+)\]\]\([^\)]+\)', '', text)  # [[212]](url)
    # 处理多个引用 [212][213][214]
    text = re.sub(r'\[(\d+)\](?:\[(\d+)\])*', '', text)
    
    # Remove HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&mdash;', '—')
    text = text.replace('&ndash;', '–')
    text = text.replace('&hellip;', '...')
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


def parse_date_from_text(text: str) -> Optional[tuple]:
    """
    Parses dates from text, returning (month, day) or None
    Supported formats: **November 5**, **November 5 “...”**, etc.
    Fixed: Uses more precise regular expressions
    """
    # Matches **Month Day** format, allowing arbitrary content in between (including quotes, parentheses, etc.)
    # Uses non-greedy matching to capture content between the first ** and last **
    date_pattern = r'\*\*([A-Za-z]+)\s+(\d+).*?\*\*'
    match = re.search(date_pattern, text)
    
    if match:
        month_str = match.group(1)
        day = int(match.group(2))
        
        
        month_map = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        
        month = month_map.get(month_str.lower())
        if month:
            return (month, day)
    
    return None


def parse_from_markdown(markdown: str, base_url: str) -> Optional[Dict]:
    """
    Parse 2024 timeline from Markdown (pure text only)
    Extract up to November 3, 2024 (excluding November 4 and later)
    """
    lines = markdown.split('\n')
    
    # Find "2024" section
    section_start = -1
    for i, line in enumerate(lines):
        if re.match(r'^#{2,}\s+2024\b', line):
            section_start = i
            break
    
    if section_start == -1:
        return None
    
    # Extract section title
    title_line = lines[section_start]
    section_title = re.sub(r'^#+\s*', '', title_line).strip()
    
    # Extract all content under 2024, until next year (2025) or November 4
    content_lines = []
    subsections = []
    current_subsection = None
    paragraphs = []
    current_month = None
    
    for i in range(section_start + 1, len(lines)):
        line = lines[i]
        
        # Stop at next year (2025)
        if re.match(r'^#{2,}\s+2025\b', line):
            break
        
        # Stop at next top-level section
        if line.startswith('##') and not line.startswith('###'):
            if re.search(r'\b(2025|2026|2027|2028|2029|2030)\b', line):
                break
        
        # Process month subsections
        month_pattern = r'^###\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+2024'
        if re.match(month_pattern, line, re.IGNORECASE):
            if current_subsection:
                subsections.append(current_subsection)
            
            month_title = re.sub(r'^#+\s*', '', line).strip()
            month_title = clean_text(month_title)
            current_month = month_title
            
            current_subsection = {
                "level": 3,
                "title": month_title,
                "content": "",
                "paragraphs": []
            }
            content_lines.append(month_title)
        
        # Process other level 3 headings
        elif line.startswith('###') and not line.startswith('####'):
            if current_subsection:
                subsections.append(current_subsection)
            
            title = re.sub(r'^#+\s*', '', line).strip()
            title = clean_text(title)
            current_subsection = {
                "level": 3,
                "title": title,
                "content": "",
                "paragraphs": []
            }
            content_lines.append(title)
        
        # Process normal content
        else:
            if line.strip():
                # 关键逻辑：如果是11月，检查日期
                if current_month and 'november' in current_month.lower():
                    date_info = parse_date_from_text(line)
                    if date_info:
                        month, day = date_info
                        # 如果遇到November 4或更大的日期，停止
                        if month == 11 and day >= 4:
                            # 停止处理，不包含这一行
                            break
                
                # Clean text
                cleaned_line = clean_text(line)
                
                if cleaned_line:
                    content_lines.append(cleaned_line)
                    
                    # Extract paragraphs
                    if not cleaned_line.startswith('#') and not cleaned_line.startswith('*') and not cleaned_line.startswith('-'):
                        paragraphs.append(cleaned_line)
                    
                    if current_subsection:
                        current_subsection["content"] += cleaned_line + "\n"
                        if not cleaned_line.startswith('#'):
                            current_subsection["paragraphs"].append(cleaned_line)
    
    # Add last subsection
    if current_subsection:
        subsections.append(current_subsection)
    
    # Merge all content
    main_text = '\n'.join(content_lines)
    
    if not main_text.strip():
        return None
    
    return {
        "section_id": "2024",
        "section_title": clean_text(section_title),
        "url": f"{base_url}#2024",
        "extracted_at": datetime.now().isoformat(),
        "content": {
            "main_text": main_text,
            "subsections": subsections
        },
        "metadata": {
            "word_count": len(main_text.split()),
            "paragraph_count": len(paragraphs),
            "subsection_count": len(subsections)
        }
    }


def parse_from_html(html: str, base_url: str) -> Optional[Dict]:
    """
    Parse 2024 timeline from HTML (pure text only)
    Extract up to November 3, 2024 (excluding November 4 and later)
    """
    soup = BeautifulSoup(html, 'html.parser')
    
  
    section_span = soup.find('span', {'id': '2024'})
    if not section_span:
        section_span = soup.find('span', class_='mw-headline', id='2024')
    
    if not section_span:
        h2_elements = soup.find_all('h2')
        for h2 in h2_elements:
            h2_text = clean_text(h2.get_text()).strip()
            if h2_text == '2024':
                section_span = h2.find('span', {'id': '2024'})
                if not section_span:
                    section_span = h2.find('span', class_='mw-headline')
                if section_span:
                    break
    
    if not section_span:
        return None
    
    # Find parent h2
    h2_element = section_span.find_parent('h2')
    if not h2_element:
        h2_element = section_span if section_span.name == 'h2' else None
    
    if not h2_element:
        return None
    
    section_title = clean_text(h2_element.get_text())
    
    parser_output = soup.find('div', class_='mw-parser-output')
    if not parser_output:
        return None
    
    all_elements = parser_output.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol'], recursive=False)
    h2_index = None
    
    for i, elem in enumerate(all_elements):
        if elem == h2_element or elem.find('span', {'id': '2024'}):
            h2_index = i
            break
    
    if h2_index is None:
        return None
    
    content_elements = []
    for i in range(h2_index + 1, len(all_elements)):
        elem = all_elements[i]
        if elem.name == 'h2':
            text = clean_text(elem.get_text())
            if re.search(r'\b(2025|2026|2027|2028|2029|2030)\b', text):
                break
            span = elem.find('span', {'id': True})
            if span and span.get('id') in ['2025', '2026', '2027', '2028', '2029', '2030']:
                break
        content_elements.append(elem)
    
    # Parse content with date filtering
    subsections = []
    paragraphs = []
    current_subsection = None
    current_month = None
    
    for element in content_elements:
        if element.name in ['h3', 'h4']:
            if current_subsection:
                subsections.append(current_subsection)
            
            title = clean_text(element.get_text())
            current_month = title
            current_subsection = {
                "level": int(element.name[1]),
                "title": title,
                "content": "",
                "paragraphs": []
            }
        
        elif element.name == 'p':
            text = clean_text(element.get_text())
            if text:
                
                if current_month and 'november' in current_month.lower():
                    date_info = parse_date_from_text(text)
                    if date_info:
                        month, day = date_info
                        # If encountering November 4 or a later date, stop
                        if month == 11 and day >= 4:
                            break
                
                paragraphs.append(text)
                if current_subsection:
                    current_subsection["paragraphs"].append(text)
                    current_subsection["content"] += text + "\n\n"
        
        elif element.name in ['ul', 'ol']:
            items = []
            for li in element.find_all('li', recursive=False):
                item_text = clean_text(li.get_text())
                if item_text:
                    # Key Logic: If it's November, check the date
                    if current_month and 'november' in current_month.lower():
                        date_info = parse_date_from_text(item_text)
                        if date_info:
                            month, day = date_info
                    
                            if month == 11 and day >= 4:
                                break
                    
                    items.append(item_text)
                    if current_subsection:
                        current_subsection["content"] += f"- {item_text}\n"
            
            if current_subsection and items:
                current_subsection["paragraphs"].extend(items)
    
    if current_subsection:
        subsections.append(current_subsection)
    
    main_text = "\n\n".join(paragraphs)
    
    if not main_text.strip():
        return None
    
    return {
        "section_id": "2024",
        "section_title": section_title,
        "url": f"{base_url}#2024",
        "extracted_at": datetime.now().isoformat(),
        "content": {
            "main_text": main_text,
            "subsections": subsections
        },
        "metadata": {
            "word_count": len(main_text.split()),
            "paragraph_count": len(paragraphs),
            "subsection_count": len(subsections)
        }
    }


def parse_timeline_2024(html: str, markdown: str = None, base_url: str = "") -> Optional[Dict]:
    """
    Parse 2024 timeline section (pure text only, no links)
    Extract up to November 3, 2024 (excluding November 4 and later)
    Priority: Markdown > HTML
    """
    # Method 1: Try Markdown first
    if markdown and len(markdown) > 100:
        markdown_data = parse_from_markdown(markdown, base_url)
        if markdown_data and markdown_data.get('content', {}).get('main_text'):
            print("   ✅ Using Markdown parsing")
            return markdown_data
    
    # Method 2: Fallback to HTML
    print("   ⚠️ Markdown parsing failed, trying HTML parsing")
    html_data = parse_from_html(html, base_url)
    if html_data:
        return html_data
    
    return None


def format_timeline_as_text(data: Dict) -> str:
    """
    Format timeline data as plain text (for LLM)
    """
    output = f"# {data['section_title']}\n\n"
    output += f"**Source**: {data['url']}\n"
    output += f"**Extracted**: {data['extracted_at']}\n"
    output += f"**Date Range**: January 1, 2024 - November 3, 2024 (excluding November 4 and later)\n\n"
    
    output += f"## Overview\n"
    output += f"- Total words: {data['metadata']['word_count']:,}\n"
    output += f"- Paragraphs: {data['metadata']['paragraph_count']}\n"
    output += f"- Months covered: {data['metadata']['subsection_count']}\n\n"
    
    output += f"## Timeline Content\n\n"
    
    for subsection in data['content']['subsections']:
        level_marker = "#" * (subsection['level'] + 1)
        output += f"{level_marker} {subsection['title']}\n\n"
        
        for para in subsection['paragraphs']:
            if para and len(para) > 20:
                output += f"{para}\n\n"
    
    return output


def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description="Extract 2024 timeline from Wikipedia (up to November 3, 2024)")
    parser.add_argument("url", nargs="?", 
                       default="https://en.wikipedia.org/wiki/Timeline_of_the_2024_United_States_presidential_election",
                       help="Wikipedia timeline page URL")
    parser.add_argument("-o", "--output-dir", default="wiki_results",
                       help="Output directory (default: wiki_results)")
    parser.add_argument("-p", "--prefix", default="timeline_2024",
                       help="Output file prefix (default: timeline_2024)")
    parser.add_argument("--show-browser", action="store_true",
                       help="Show browser window (not headless)")
    parser.add_argument("--save-html", action="store_true",
                       help="Save HTML file for debugging")
    
    args = parser.parse_args()
    
    # Run extraction
    result = asyncio.run(extract_timeline_2024(
        url=args.url,
        output_dir=args.output_dir,
        output_prefix=args.prefix,
        headless=not args.show_browser,
        verbose=True,
        save_html=args.save_html
    ))
    
    if result.get("success"):
        print("\n✅ Extraction successful!")
        if "files" in result:
            print(f"📄 JSON: {result['files']['json']}")
            print(f"📝 Text: {result['files']['txt']}")
    else:
        print(f"\n❌ Extraction failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()