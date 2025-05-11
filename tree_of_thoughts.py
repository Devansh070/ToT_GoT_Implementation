from typing import List, Dict, Any
from dataclasses import dataclass
import json
import os
import matplotlib.pyplot as plt
import networkx as nx

# Import PDF parsing utilities
from pdf_to_tree_of_thoughts import pdf_to_tree_of_thoughts

@dataclass
class Thought:
    content: str
    score: float = 0.0
    children: List['Thought'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class TreeOfThoughts:
    def __init__(self):
        """Initialize TreeOfThoughts with a free LLM, fallback to simulated."""
        self.use_llm = False
        # inside your class

        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            self.model_name = "google/flan-t5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if hasattr(self.model, "cuda") and self.model.cuda else -1
            )
            self.use_llm = True
            # The same T5 Flan Small model is used for both agentic response and critic LLM
            print("Loaded free LLM model for generation and critic (T5 Flan Small).")
        except Exception as e:
            print(f"Could not load free LLM model, using simulated model. Reason: {e}")
            self.generator = None

    def _extract_paper_details(self, paper_data: Dict) -> str:
        """Extract and format more details from the parsed paper tree."""
        if not isinstance(paper_data, dict):
            return "No details available."
        root = paper_data.get("root", {})
        children = root.get("children", [])
        details = []
        for section in children[:5]:  # Show up to 5 sections for brevity
            section_title = section.get("section", "N/A")
            content = section.get("content", "")
            snippet = content[:120].replace('\n', ' ') + ("..." if len(content) > 120 else "")
            details.append(f"Section {section_title}: {snippet}")
        return "\n".join(details) if details else "No sections found."

    def _generate_agentic_response(self, prompt: str, paper_data: Dict) -> List[str]:
        """Generate multiple responses using LLM or simulate."""
        paper_details = self._extract_paper_details(paper_data)
        full_prompt = (
            "As an expert academic reviewer, analyze this research paper.\n"
            "Consider originality, methodology, significance, and potential impact.\n"
            "Provide a detailed analysis focusing on these aspects.\n\n"
            f"Paper details:\n{paper_details}\n\n{prompt}"
        )
        if self.use_llm and self.generator:
            try:
                responses = []
                for _ in range(3):
                    output = self.generator(full_prompt, max_length=256, num_return_sequences=1, temperature=0.7, do_sample=True)
                    responses.append(output[0]['generated_text'].strip())
                return responses
            except Exception as e:
                print(f"LLM generation failed, falling back to simulated: {e}")
        # Simulated fallback
        return [
            f"Simulated analysis 1 for: {paper_details[:40]}...",
            f"Simulated analysis 2 for: {paper_details[:40]}...",
            f"Simulated analysis 3 for: {paper_details[:40]}..."
        ]

    def _evaluate_critic(self, thoughts: List[str], paper_data: Dict) -> List[float]:
        """Evaluate thoughts using LLM or simulate."""
        if self.use_llm and self.generator:
            try:
                scores = []
                for thought in thoughts:
                    prompt = (
                        "Rate the following analysis on a scale of 0 to 1, considering:\n"
                        "- Depth of analysis\n"
                        "- Supporting evidence\n"
                        "- Logical reasoning\n"
                        "- Consideration of limitations\n"
                        "Provide only the numerical score.\n\n"
                        f"Analysis:\n{thought}"
                    )
                    output = self.generator(prompt, max_length=8, num_return_sequences=1, temperature=0.3)
                    score_text = output[0]['generated_text'].strip()
                    try:
                        score = float(''.join(c for c in score_text if c.isdigit() or c == '.'))
                        scores.append(min(max(score, 0), 1))
                    except Exception:
                        scores.append(0.5)
                return scores
            except Exception as e:
                print(f"LLM critic failed, falling back to simulated: {e}")
        # Simulated fallback
        return [0.8, 0.6, 0.7][:len(thoughts)]

    def _make_final_decision(self, thought_tree: Thought, paper_data: Dict) -> Dict[str, Any]:
        """
        Make final decision about paper publishability.
        Uses critic LLM if available, otherwise simulated.
        """
        paper_details = self._extract_paper_details(paper_data)
        if self.use_llm and self.generator:
            try:
                prompt = (
                    "Based on the provided analysis tree and paper details, determine:\n"
                    "1. Whether the paper is publishable (yes/no). You can lower the threshold\n"
                    "2. Confidence score (0-1)\n"
                    "3. Key reasons for the decision\n"
                    "Format: publishable: yes/no\nconfidence: [score]\nreasons: [text]\n\n"
                    f"Analysis tree:\n{self._format_thought_tree(thought_tree)}\n\nPaper details:\n{paper_details}"
                )
                output = self.generator(prompt, max_length=128, num_return_sequences=1, temperature=0.3)
                decision_text = output[0]['generated_text'].strip()
                lines = decision_text.split('\n')
                publishable = any(line.lower().startswith('publishable: yes') for line in lines)
                confidence = 0.8
                for line in lines:
                    if line.lower().startswith('confidence:'):
                        try:
                            confidence = float(line.split(':')[1].strip())
                            confidence = min(max(confidence, 0), 1)
                        except Exception:
                            pass
                return {
                    "publishable": publishable,
                    "confidence": confidence,
                    "reasons": decision_text
                }
            except Exception as e:
                print(f"LLM decision failed, falling back to simulated: {e}")
        # Simulated fallback
        return {
            "publishable": True,
            "confidence": 0.95,
            "reasons": f"Simulated: The paper meets all criteria.\nDetails:\n{paper_details}"
        }

    def _format_thought_tree(self, thought: Thought, level: int = 0) -> str:
        indent = "  " * level
        result = f"{indent}Thought (score={thought.score:.2f}): {thought.content}\n"
        for child in thought.children:
            result += self._format_thought_tree(child, level + 1)
        return result

    def _thought_tree_to_dict(self, thought: 'Thought') -> dict:
        """Recursively convert the Thought tree to a dictionary."""
        return {
            "content": thought.content,
            "score": thought.score,
            "children": [self._thought_tree_to_dict(child) for child in thought.children]
        }

    def export_tree_png(self, thought: 'Thought', output_path: str = None):
        """
        Export the thought tree as a PNG image, saved in a 'tht_png_exports' folder.
        This function works in non-GUI (headless) environments by using the 'Agg' backend.
        The PNG is saved to disk only; no image data is returned or embedded in JSON.
        If pygraphviz is not available, uses spring layout.
        """
        import matplotlib
        matplotlib.use('Agg')  # Ensure non-GUI backend
        import matplotlib.pyplot as plt
        import networkx as nx
        from pathlib import Path

        def add_nodes_edges(G, node, parent_id=None, node_id=[0]):
            current_id = node_id[0]
            G.add_node(current_id, label=f"{node.content[:30]} ({node.score:.2f})")
            if parent_id is not None:
                G.add_edge(parent_id, current_id)
            for child in node.children:
                node_id[0] += 1
                add_nodes_edges(G, child, current_id, node_id)

        G = nx.DiGraph()
        add_nodes_edges(G, thought)

        # Try pygraphviz layout, fallback to spring layout if not available
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception as e:
            print("pygraphviz not available or failed, using spring layout instead.")
            pos = nx.spring_layout(G)

        export_dir = Path("tht_png_exports")
        export_dir.mkdir(exist_ok=True)

        if not output_path:
            output_path = export_dir / "thought_tree.png"
        else:
            output_path = export_dir / output_path

        labels = nx.get_node_attributes(G, 'label')
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, labels=labels,
                node_size=2000, node_color='lightblue', font_size=8, arrows=True)
        plt.tight_layout()
        plt.savefig(str(output_path))
        plt.close()
        print(f"Graph saved to {output_path}")

    def analyze_paper(self, paper_data: Dict, graph_summary: Dict) -> Dict[str, Any]:
        """Main method to analyze a paper using Tree of Thoughts approach."""
        root = Thought("Initial Analysis")
        full_context = {
            "paper_data": paper_data,
            "graph_summary": graph_summary
        }
        current_level = [root]
        for cycle in range(3):
            next_level = []
            for parent in current_level:
                thoughts = self._generate_agentic_response(
                    f"Cycle {cycle + 1}: Analyze the paper considering previous thoughts:\n{self._format_thought_tree(root)}",
                    paper_data
                )
                scores = self._evaluate_critic(thoughts, paper_data)
                thought_nodes = [
                    Thought(content=thought, score=score)
                    for thought, score in zip(thoughts, scores)
                ]
                thought_nodes.sort(key=lambda x: x.score, reverse=True)
                selected_thoughts = thought_nodes[:2]
                parent.children.extend(selected_thoughts)
                next_level.extend(selected_thoughts)
            current_level = next_level
        decision = self._make_final_decision(root, paper_data)
        return {
            "decision": decision,
            "thought_tree": self._thought_tree_to_dict(root)
        }

def save_result_to_json(result: Any, output_path: str):
    """Save the result dictionary to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {os.path.abspath(output_path)}")

def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Analyze PDFs in a folder using Tree of Thoughts (LLM with simulated fallback)")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="tree_of_thoughts_result.json", help="Output JSON file")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f"Provided path {pdf_dir} is not a valid directory.")
        return

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    graph_summary = {}

    tot = TreeOfThoughts()
    results = []
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        tree = pdf_to_tree_of_thoughts(str(pdf_path))
        result = tot.analyze_paper(tree, graph_summary)
        results.append({
            "file": str(pdf_path),
            "result": result
        })

    save_result_to_json(results, args.output)

if __name__ == "__main__":
    main()