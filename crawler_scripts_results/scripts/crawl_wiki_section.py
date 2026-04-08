"""
Wikipedia section extractor script - based on Crawl4AI
Specialized for extracting specific sections of Wikipedia pages
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


async def extract_wiki_section(
    url: str,
    section_id: str,
    output_dir: str = "wiki_results",
    output_prefix: str = None,
    headless: bool = True,
    verbose: bool = True,
    save_html: bool = False
):
    """
    Extract specific sections of Wikipedia pages
    
    Parameters:
        url: Wikipedia page URL (without anchor)
        section_id: Section ID (e.g., "Campaign_issues")
        output_dir: Output directory
        output_prefix: Output file name prefix
        headless: Whether to use headless mode
        verbose: Whether to show detailed logs
        save_html: Whether to save HTML (for debugging)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate output file name prefix
    if not output_prefix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"wiki_{section_id}_{timestamp}"
    
    # Build full URL (with anchor)
    full_url = f"{url}#{section_id}"
    
    print(f" Starting to extract Wikipedia section")
    print(f" Page: {url}")
    print(f" Section: {section_id}")
    print(f" Output directory: {output_path.absolute()}")
    print(f" File prefix: {output_prefix}")
    print("-" * 60)
    
    # Configure browser
    browser_config = BrowserConfig(
        headless=headless,
        verbose=verbose
    )
    
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Configure crawling parameters - locate to specific section
            config = CrawlerRunConfig(
                wait_for=None,  # Do not wait for specific element, avoid timeout
                css_selector=None,  # First get the full page, then parse
                delay_before_return_html=3.0,  # Wait for page load
                word_count_threshold=1,  # Keep all text
                excluded_tags=['nav', 'header', 'footer', 'script', 'style'],
                cache_mode="bypass"
            )
            
            print(" Step 1: Load page...")
            result = await crawler.arun(
                url=full_url,
                config=config
            )
            
            if not result.success:
                error_msg = result.error_message if hasattr(result, 'error_message') else 'Unknown error'
                print(f" Page load failed: {error_msg}")
                return {"success": False, "error": "Page load failed"}
            
            print(" Page load successful")
            
            # Save HTML (for debugging)
            if save_html:
                html_file = output_path / f"{output_prefix}.html"
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(result.html)
                print(f" HTML saved: {html_file.name}")
            
            # Parse section content
            print(f"\n Step 2: Parse section content...")
            structured_data = parse_wiki_section(
                result.html,
                result.markdown,
                section_id,
                url
            )
            
            if not structured_data:
                print(" Section not found")
                return {"success": False, "error": "Section not found"}
            
            print(f" Section content extracted successfully")
            print(f"   - Title: {structured_data.get('section_title', 'N/A')}")
            print(f"   - Subsection count: {len(structured_data.get('content', {}).get('subsections', []))}")
            print(f"   - Link count: {len(structured_data.get('content', {}).get('links', []))}")
            
            # Save JSON
            json_file = output_path / f"{output_prefix}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
            print(f"\n JSON saved: {json_file.name}")
            
            # Show preview
            print("-" * 60)
            print(" Content preview:")
            preview_text = structured_data.get('content', {}).get('main_text', '')[:300]
            print(preview_text)
            if len(structured_data.get('content', {}).get('main_text', '')) > 300:
                print("...")
            
            return {
                "success": True,
                "data": structured_data,
                "files": {
                    "json": str(json_file),
                    "html": str(html_file) if save_html else None
                }
            }
    
    except Exception as e:
        print(f" Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

def parse_wiki_section(html: str, markdown: str, section_id: str, base_url: str) -> Optional[Dict]:
    """
    Parse Wikipedia section content into structured data
    Priority: use Markdown, fallback to HTML
    """
    # Method 1: Use Markdown (more reliable)
    if markdown and len(markdown) > 100:  # Ensure markdown has content
        markdown_data = parse_from_markdown(markdown, section_id, base_url)
        if markdown_data and markdown_data.get('content', {}).get('main_text'):
            print(" Successfully parsed using Markdown")
            return markdown_data
    
    # Method 2: Use HTML parsing (fallback)
    print(" Markdown parsing failed, trying HTML parsing")
    html_data = parse_from_html(html, section_id, base_url)
    if html_data:
        return html_data
    
    return None


def parse_from_markdown(markdown: str, section_id: str, base_url: str) -> Optional[Dict]:
    """
    Parse section content from Markdown
    """
    import re
    
    lines = markdown.split('\n')
    
    # Find section title
    section_title = section_id.replace('_', ' ')
    section_start = -1
    
    # Find title lines containing section title (## or ###)
    for i, line in enumerate(lines):
        # Match ## Campaign issues or similar format
        if re.match(r'^#{2,}\s+' + re.escape(section_title), line, re.IGNORECASE):
            section_start = i
            break
        # Or title containing Campaign issues
        if line.startswith('##') and section_title.lower() in line.lower():
            section_start = i
            break
    
    if section_start == -1:
        return None
    
    # Extract section title
    title_line = lines[section_start]
    section_title_actual = re.sub(r'^#+\s*', '', title_line).strip()
    
    # Extract all content under the section, until the next level 2 heading（##）
    content_lines = []
    subsections = []
    links = []
    current_subsection = None
    paragraphs = []
    
    for i in range(section_start + 1, len(lines)):
        line = lines[i]
        
        # If next level 2 heading（##），stop
        if line.startswith('##') and not line.startswith('###'):
            break
        
        # Process level 3 heading (subsection)
        if line.startswith('###') and not line.startswith('####'):
            if current_subsection:
                subsections.append(current_subsection)
            
            title = re.sub(r'^#+\s*', '', line).strip()
            current_subsection = {
                "level": 3,
                "title": title,
                "content": "",
                "paragraphs": [],
                "links": []
            }
            content_lines.append(line)
        
        # Process level 4 heading
        elif line.startswith('####'):
            if current_subsection:
                subsections.append(current_subsection)
            
            title = re.sub(r'^#+\s*', '', line).strip()
            current_subsection = {
                "level": 4,
                "title": title,
                "content": "",
                "paragraphs": [],
                "links": []
            }
            content_lines.append(line)
        
        # Process normal content
        else:
            if line.strip():
                content_lines.append(line)
                
                # Extract paragraphs
                if line.strip() and not line.startswith('#') and not line.startswith('*') and not line.startswith('-'):
                    paragraphs.append(line.strip())
                
                if current_subsection:
                    current_subsection["content"] += line + "\n"
                    if line.strip() and not line.startswith('#'):
                        current_subsection["paragraphs"].append(line.strip())
                
                # Extract links (Markdown format: [text](url))
                link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
                for match in re.finditer(link_pattern, line):
                    link_text = match.group(1)
                    link_url = match.group(2)
                    
                    # Build full URL
                    if link_url.startswith('/'):
                        full_url = f"https://en.wikipedia.org{link_url}"
                    elif link_url.startswith('http'):
                        full_url = link_url
                    elif link_url.startswith('#'):
                        full_url = f"{base_url}{link_url}"
                    else:
                        full_url = f"{base_url}/{link_url}"
                    
                    link_info = {
                        "text": link_text,
                        "url": full_url,
                        "href": link_url,
                        "type": "internal" if '/wiki/' in link_url or link_url.startswith('#') else "external"
                    }
                    
                    # Avoid duplicates
                    if not any(l.get('text') == link_text and l.get('href') == link_url for l in links):
                        links.append(link_info)
                        if current_subsection:
                            current_subsection["links"].append(link_info)
    
    # Add last subsection
    if current_subsection:
        subsections.append(current_subsection)
    
    # Merge all content
    main_text = '\n'.join(content_lines)
    
    if not main_text.strip():
        return None
    
    return {
        "section_id": section_id,
        "section_title": section_title_actual,
        "url": f"{base_url}#{section_id}",
        "extracted_at": datetime.now().isoformat(),
        "content": {
            "main_text": main_text,
            "subsections": subsections,
            "links": links
        },
        "metadata": {
            "word_count": len(main_text.split()),
            "paragraph_count": len(paragraphs),
            "link_count": len(links),
            "subsection_count": len(subsections)
        }
    }


def parse_from_html(html: str, section_id: str, base_url: str) -> Optional[Dict]:
    """
    Parse section content from HTML (fallback)
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find span#Campaign_issues
    section_span = soup.find('span', {'id': section_id})
    if not section_span:
        return None
    
    # Find parent element h2
    h2_element = section_span.find_parent('h2')
    if not h2_element:
        return None
    
    # Get section title
    section_title = h2_element.get_text().strip()
    
    # Find mw-parser-output container
    parser_output = soup.find('div', class_='mw-parser-output')
    if not parser_output:
        return None
    
    # Find position of h2 in container
    all_elements = parser_output.find_all(['h2', 'h3', 'h4', 'h5', 'p', 'ul', 'ol'], recursive=False)
    h2_index = None
    
    for i, elem in enumerate(all_elements):
        if elem == h2_element or elem.find('span', {'id': section_id}):
            h2_index = i
            break
    
    if h2_index is None:
        return None
    
    # Extract all elements after h2, until the next h2
    content_elements = []
    for i in range(h2_index + 1, len(all_elements)):
        elem = all_elements[i]
        if elem.name == 'h2':
            break
        content_elements.append(elem)
    
    # Parse content
    subsections = []
    links = []
    paragraphs = []
    current_subsection = None
    
    for element in content_elements:
        if element.name in ['h3', 'h4', 'h5']:
            if current_subsection:
                subsections.append(current_subsection)
            
            current_subsection = {
                "level": int(element.name[1]),
                "title": element.get_text().strip(),
                "content": "",
                "paragraphs": [],
                "links": []
            }
        
        elif element.name == 'p':
            text = element.get_text().strip()
            if text:
                paragraphs.append(text)
                if current_subsection:
                    current_subsection["paragraphs"].append(text)
                    current_subsection["content"] += text + "\n\n"
        
        elif element.name in ['ul', 'ol']:
            items = []
            for li in element.find_all('li', recursive=False):
                item_text = li.get_text().strip()
                if item_text:
                    items.append(item_text)
                    if current_subsection:
                        current_subsection["content"] += f"- {item_text}\n"
            
            if current_subsection and items:
                current_subsection["paragraphs"].extend(items)
        
        # Extract links
        for link in element.find_all('a', href=True):
            link_text = link.get_text().strip()
            link_href = link.get('href', '')
            
            if not link_text:
                continue
            
            # Build full URL
            if link_href.startswith('/'):
                full_link_url = f"https://en.wikipedia.org{link_href}"
            elif link_href.startswith('http'):
                full_link_url = link_href
            elif link_href.startswith('#'):
                full_link_url = f"{base_url}{link_href}"
            else:
                full_link_url = f"{base_url}/{link_href}"
            
            link_info = {
                "text": link_text,
                "url": full_link_url,
                "href": link_href,
                "type": "internal" if link_href.startswith('/wiki/') or link_href.startswith('#') else "external"
            }
            
            # Avoid duplicates
            if not any(l.get('text') == link_text and l.get('href') == link_href for l in links):
                links.append(link_info)
                if current_subsection:
                    current_subsection["links"].append(link_info)
    
    # Add last subsection
    if current_subsection:
        subsections.append(current_subsection)
    
    # Merge all paragraphs text
    main_text = "\n\n".join(paragraphs)
    
    if not main_text.strip():
        return None
    
    return {
        "section_id": section_id,
        "section_title": section_title,
        "url": f"{base_url}#{section_id}",
        "extracted_at": datetime.now().isoformat(),
        "content": {
            "main_text": main_text,
            "subsections": subsections,
            "links": links
        },
        "metadata": {
            "word_count": len(main_text.split()),
            "paragraph_count": len(paragraphs),
            "link_count": len(links),
            "subsection_count": len(subsections)
        }
    }

'''
def parse_wiki_section(html: str, markdown: str, section_id: str, base_url: str) -> Optional[Dict]:
    """
    Parse Wikipedia section content into structured data
    
    Parameters:
        html: HTML content
        markdown: Markdown content
        section_id: Section ID
        base_url: Base URL
    
    Returns:
        Dict: Structured section data
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find section title element
    section_heading = soup.find('span', {'id': section_id})
    if not section_heading:
        # Try other ways to find
        section_heading = soup.find('h2', string=re.compile(section_id.replace('_', ' '), re.I))
        if not section_heading:
            section_heading = soup.find('h2', id=section_id)
    
    if not section_heading:
        return None
    
    # Get section title
    section_title = section_heading.get_text().strip() if section_heading else section_id.replace('_', ' ')
    
    # Find parent container of section (usually the parent element of h2)
    # Wikipedia structure usually is: h2 > span#section_id
    h2_element = section_heading.find_parent('h2') if section_heading.name != 'h2' else section_heading
    
    # Extract all content under the section, until the next h2
    content_elements = []
    current = h2_element.next_sibling if h2_element else None
    
    while current:
        if current.name == 'h2':
            # If next h2, stop
            break
        if hasattr(current, 'name') and current.name:
            content_elements.append(current)
        current = current.next_sibling
    
    # Parse content
    subsections = []
    links = []
    paragraphs = []
    current_subsection = None
    
    for element in content_elements:
        if element.name in ['h3', 'h4', 'h5']:
            # Save previous subsection
            if current_subsection:
                subsections.append(current_subsection)
            
            # Start new subsection
            current_subsection = {
                "level": int(element.name[1]),  # h3 -> 3
                "title": element.get_text().strip(),
                "content": "",
                "paragraphs": [],
                "links": [],
                "subsections": []
            }
        
        elif element.name == 'p':
            text = element.get_text().strip()
            if text:
                paragraphs.append(text)
                if current_subsection:
                    current_subsection["paragraphs"].append(text)
                    current_subsection["content"] += text + "\n\n"
                else:
                    paragraphs.append(text)
        
        elif element.name in ['ul', 'ol']:
            items = []
            for li in element.find_all('li', recursive=False):
                item_text = li.get_text().strip()
                items.append(item_text)
                if current_subsection:
                    current_subsection["content"] += f"- {item_text}\n"
            
            if current_subsection:
                current_subsection["paragraphs"].extend(items)
        
        # Extract links
        for link in element.find_all('a', href=True):
            link_text = link.get_text().strip()
            link_href = link.get('href', '')
            
            # Build full URL
            if link_href.startswith('/'):
                full_link_url = f"https://en.wikipedia.org{link_href}"
            elif link_href.startswith('http'):
                full_link_url = link_href
            else:
                full_link_url = f"{base_url}{link_href}"
            
            link_info = {
                "text": link_text,
                "url": full_link_url,
                "href": link_href,
                "type": "internal" if link_href.startswith('/wiki/') or link_href.startswith('#') else "external"
            }
            
            if link_text and link_info not in links:
                links.append(link_info)
                if current_subsection:
                    current_subsection["links"].append(link_info)
    
    # Add last subsection
    if current_subsection:
        subsections.append(current_subsection)
    
    # Merge all paragraphs text
    main_text = "\n\n".join(paragraphs)
    
    # Build structured data
    structured_data = {
        "section_id": section_id,
        "section_title": section_title,
        "url": f"{base_url}#{section_id}",
        "extracted_at": datetime.now().isoformat(),
        "content": {
            "main_text": main_text,
            "subsections": subsections,
            "links": links
        },
        "metadata": {
            "word_count": len(main_text.split()),
            "paragraph_count": len(paragraphs),
            "link_count": len(links),
            "subsection_count": len(subsections)
        }
    }
    
    return structured_data
'''

def main():
    """Command-Line Entry"""
    parser = argparse.ArgumentParser(
        description="Wikipedia section extractor tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example usage:
        python crawl_wiki_section.py https://en.wikipedia.org/wiki/2024_United_States_presidential_election Campaign_issues
        python crawl_wiki_section.py https://en.wikipedia.org/wiki/2024_United_States_presidential_election Campaign_issues -o ./output -p my_section
        python crawl_wiki_section.py https://en.wikipedia.org/wiki/2024_United_States_presidential_election Campaign_issues --save-html
        """
    )
    
    parser.add_argument("url", help="Wikipedia page URL (without anchor)")
    parser.add_argument("section_id", help="Section ID (e.g., Campaign_issues)")
    parser.add_argument("-o", "--output-dir", default="wiki_results", help="Output directory (default: wiki_results)")
    parser.add_argument("-p", "--prefix", help="Output file name prefix (default: auto-generated)")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window")
    parser.add_argument("--save-html", action="store_true", help="Save HTML file")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    result = asyncio.run(extract_wiki_section(
        url=args.url,
        section_id=args.section_id,
        output_dir=args.output_dir,
        output_prefix=args.prefix,
        headless=not args.no_headless,
        verbose=not args.quiet,
        save_html=args.save_html
    ))
    
    if result["success"]:
        print(f"\n Extraction completed!")
        print(f" Statistics:")
        if "data" in result:
            metadata = result["data"].get("metadata", {})
            print(f"   - Word count: {metadata.get('word_count', 0):,}")
            print(f"   - Paragraph count: {metadata.get('paragraph_count', 0)}")
            print(f"   - Link count: {metadata.get('link_count', 0)}")
            print(f"   - Subsection count: {metadata.get('subsection_count', 0)}")
        sys.exit(0)
    else:
        print(f"\n Extraction failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()