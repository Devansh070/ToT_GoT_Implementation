import os
from pathlib import Path
from typing import List, Dict, Any
from pdf_to_tree_of_thoughts import pdf_to_tree_of_thoughts

def parse_papers_from_dir(pdf_dir: str) -> List[Dict[str, Any]]:
    """
    Parse all PDFs in a directory and return a list of dicts with file path and tree_of_thoughts.
    """
    pdf_dir_path = Path(pdf_dir)
    papers = []
    for pdf_path in pdf_dir_path.glob("*.pdf"):
        try:
            tree = pdf_to_tree_of_thoughts(str(pdf_path))
            papers.append({
                "file_path": str(pdf_path),
                "tree_of_thoughts": tree
            })
        except Exception as e:
            print(f"Failed to parse {pdf_path}: {e}")
    return papers

# Example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse all PDFs in a directory to tree_of_thoughts structures.")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="parsed_papers.json", help="Output JSON file")
    args = parser.parse_args()

    papers = parse_papers_from_dir(args.pdf_dir)
    import json
    with open(args.output, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"Parsed {len(papers)} papers and saved to {args.output}")
