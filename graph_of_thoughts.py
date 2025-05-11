import networkx as nx
from datetime import datetime
from pypdf import PdfReader
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
class PaperNode:
    def __init__(self, pdf_path: str, timestamp: datetime):
        self.pdf_path = pdf_path
        self.timestamp = timestamp
        self.content = self._extract_content()
        self.metadata = self._extract_metadata()
        self.authors = self._extract_authors()
        self.citations = self._extract_citations()
        self.topics = self._extract_topics()
        
    def _extract_content(self) -> str:
        """Extract text content from PDF."""
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return text
        except Exception as e:
            print(f"Error extracting content from {self.pdf_path}: {e}")
            return ""
            
    def _extract_metadata(self) -> Dict:
        """Extract metadata from PDF."""
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PdfReader(file)
                return reader.metadata
        except Exception as e:
            print(f"Error extracting metadata from {self.pdf_path}: {e}")
            return {}

    def _extract_authors(self) -> List[str]:
        """Extract authors from metadata or content."""
        meta = self.metadata
        authors = []
        if meta and '/Author' in meta and meta['/Author']:
            raw_author = meta['/Author']
            if isinstance(raw_author, bytes):
                raw_author = raw_author.decode('utf-8', errors='ignore')
            authors = [str(raw_author).strip()]

        # Fallback: look for "Author" in content (very naive)
        if not authors:
            lines = self.content.split('\n')
            for line in lines[:10]:
                if "author" in line.lower():
                    authors = [line.strip()]
                    break
        # If still empty, use a placeholder
        if not authors:
            authors = ["Unknown"]
        return authors

    def _extract_citations(self) -> List[str]:
        """Extract citations from content."""
        citations = []
        lines = self.content.split('\n')
        in_refs = False
        for line in lines:
            if "references" in line.lower() or "citations" in line.lower():
                in_refs = True
                continue
            if in_refs:
                if line.strip() == "" or len(citations) > 20:
                    break
                citations.append(line.strip())
        return citations

    def _extract_topics(self) -> List[str]:
        """Extract topics from metadata or content."""
        meta = self.metadata
        if meta and '/Keywords' in meta:
            return [k.strip() for k in meta['/Keywords'].split(',')]
        # Fallback: use first 100 words as "topics"
        words = self.content.split()
        return list(set(words[:100]))[:5]

class GraphOfThoughts:
    def __init__(self):
        self.graph = nx.Graph()
        self.similarity_threshold = 0.7  # for redundancy
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            self.model_name = "google/flan-t5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.llm = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if hasattr(self.model, "cuda") and self.model.cuda else -1
            )
            self.use_llm = True
            print("Loaded flan-t5-small for redundancy scoring.")
        except Exception as e:
            print(f"Could not load flan-t5-small, using simulated redundancy scoring. Reason: {e}")
            self.llm = None
            self.use_llm = False

    def add_paper(self, pdf_path: str):
        """Add a paper to the graph."""
        timestamp = datetime.now()
        node = PaperNode(pdf_path, timestamp)
        self.graph.add_node(pdf_path, data=node)
        self._update_relationships(pdf_path)
        
    def _calculate_redundancy_score(self, node1: PaperNode, node2: PaperNode) -> float:
        """Calculate redundancy score between two papers."""
        if self.use_llm and self.llm:
            prompt = (
                "Given the following two research papers, rate their redundancy on a scale from 0 (not redundant) to 1 (completely redundant). "
                "Consider overlap in content, authors, citations, and topics.\n\n"
                f"Paper 1 Authors: {node1.authors}\n"
                f"Paper 1 Topics: {node1.topics}\n"
                f"Paper 1 Citations: {node1.citations[:3]}\n"
                f"Paper 1 Content: {node1.content[:300]}\n\n"
                f"Paper 2 Authors: {node2.authors}\n"
                f"Paper 2 Topics: {node2.topics}\n"
                f"Paper 2 Citations: {node2.citations[:3]}\n"
                f"Paper 2 Content: {node2.content[:300]}\n\n"
                "Redundancy score (0-1):"
            )
            try:
                output = self.llm(prompt, max_length=8, num_return_sequences=1, temperature=0.3)
                score_text = output[0]['generated_text'].strip()
                score = float(''.join(c for c in score_text if c.isdigit() or c == '.'))
                return min(max(score, 0), 1)
            except Exception as e:
                print(f"LLM redundancy scoring failed: {e}")
        # Simulated fallback: simple overlap in authors, topics, and content
        author_overlap = len(set(node1.authors) & set(node2.authors)) > 0
        topic_overlap = len(set(node1.topics) & set(node2.topics)) > 0
        content_overlap = self._simple_content_overlap(node1.content, node2.content)
        score = 0.0
        if author_overlap:
            score += 0.4
        if topic_overlap:
            score += 0.3
        if content_overlap:
            score += 0.4
        return min(score, 1.0)

    def _simple_content_overlap(self, content1: str, content2: str) -> bool:
        """Check for simple content overlap."""
        words1 = content1.split()
        words2 = content2.split()
        for i in range(0, len(words1) - 9):
            phrase = ' '.join(words1[i:i+10])
            if phrase in content2:
                return True
        for i in range(0, len(words2) - 9):
            phrase = ' '.join(words2[i:i+10])
            if phrase in content1:
                return True
        return False

    def _update_relationships(self, new_paper_path: str):
        """Update relationships between papers in the graph."""
        new_node = self.graph.nodes[new_paper_path]['data']
        for node_path in self.graph.nodes():
            if node_path != new_paper_path:
                other_node = self.graph.nodes[node_path]['data']
                # Relationship types
                relationships = []
                if set(new_node.authors) & set(other_node.authors):
                    relationships.append("author")
                if set(new_node.topics) & set(other_node.topics):
                    relationships.append("topic")
                if set(new_node.citations) & set(other_node.citations):
                    relationships.append("citation")
                redundancy_score = self._calculate_redundancy_score(new_node, other_node)
                redundant = redundancy_score > self.similarity_threshold
                self.graph.add_edge(
                    new_paper_path,
                    node_path,
                    relationships=relationships,
                    redundancy_score=redundancy_score,
                    redundant=redundant
                )

    def get_paper_relationships(self, pdf_path: str) -> List[Tuple[str, List[str], float, bool]]:
        """Get all relationships for a specific paper."""
        if pdf_path not in self.graph:
            return []
        relationships = []
        for neighbor in self.graph.neighbors(pdf_path):
            edge_data = self.graph.get_edge_data(pdf_path, neighbor)
            relationships.append((
                neighbor,
                edge_data['relationships'],
                edge_data['redundancy_score'],
                edge_data['redundant']
            ))
        return relationships

    def get_graph_summary(self) -> Dict:
        """Get a summary of the graph for LLM analysis."""
        summary = {
            'papers': {},
            'edges': [],
            'visualization': {
                'nodes': [],
                'edges': []
            }
        }
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]['data']
            summary['papers'][node] = {
                'timestamp': node_data.timestamp.isoformat(),
                'metadata': node_data.metadata,
                'authors': node_data.authors,
                'topics': node_data.topics,
                'citations': node_data.citations
            }
            summary['visualization']['nodes'].append({
                'id': node,
                'redundant': self.is_redundant(node)
            })
        for edge in self.graph.edges(data=True):
            summary['edges'].append({
                'paper1': edge[0],
                'paper2': edge[1],
                'relationships': edge[2]['relationships'],
                'redundancy_score': edge[2]['redundancy_score'],
                'redundant': edge[2]['redundant']
            })
            summary['visualization']['edges'].append({
                'source': edge[0],
                'target': edge[1],
                'redundancy_score': edge[2]['redundancy_score'],
                'redundant': edge[2]['redundant']
            })
        return summary

    def is_redundant(self, pdf_path: str) -> bool:
        """Check if a paper is redundant."""
        for neighbor in self.graph.neighbors(pdf_path):
            edge_data = self.graph.get_edge_data(pdf_path, neighbor)
            if edge_data.get('redundant', False):
                return True
        return False

    def save_graph_to_json(self, output_path: str):
        """Save graph summary to a JSON file."""
        summary = self.get_graph_summary()
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Graph summary saved to {output_path}")
        import matplotlib.pyplot as plt

    def visualize_graph(self):
        pos = nx.spring_layout(self.graph, seed=42)
        labels = {node: node.split('/')[-1] for node in self.graph.nodes}
        redundant_nodes = [n for n in self.graph.nodes if self.is_redundant(n)]
        normal_nodes = list(set(self.graph.nodes) - set(redundant_nodes))

        # Draw
        nx.draw_networkx_nodes(self.graph, pos, nodelist=normal_nodes, node_color='lightblue')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=redundant_nodes, node_color='salmon')
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray')
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8)

        plt.title("Graph of Thought (Papers as Nodes)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        plt.savefig("graph_output.png")
        print("Graph saved to graph_output.png")



# Example usage:
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Build and analyze a research paper graph.")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="graph_of_thoughts_result.json", help="Output JSON file")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    graph = GraphOfThoughts()
    for pdf_path in pdf_files:
        print(f"Adding {pdf_path.name} to graph...")
        graph.add_paper(str(pdf_path))
    graph.save_graph_to_json(args.output)
    graph.visualize_graph()
    print(f"Graph of thoughts built and saved to {args.output}.")