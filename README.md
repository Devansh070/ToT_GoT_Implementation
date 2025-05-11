# Research Paper Analysis System

This repository implements advanced techniques for research paper analysis using Tree of Thoughts and Graph of Thoughts approaches. These methods allow for both relationship-based analysis between multiple papers and in-depth evaluation of individual papers.

## Project Overview

I've developed two complementary analysis frameworks:

1. **Graph of Thoughts**: A network-based approach to analyze relationships and redundancy across multiple papers
2. **Tree of Thoughts**: A hierarchical analysis system that evaluates individual papers through recursive refinement

Both implementations leverage language models when available but include fallback mechanisms using simulated models for environments without LLM access.

## File Structure

The project consists of these key files:

- graph_of_thoughts.py (Graph of Thoughts implementation)
  - Contains GraphOfThoughts class
  - Manages paper relationship analysis
  - Includes visualization and redundancy detection

- tree_of_thoughts.py (Tree of Thoughts implementation)
  - Contains TreeOfThoughts class
  - Implements paper analysis through recursive thought generation
  - Includes thought evaluation and decision making
  
There are two separate files for parsing PDFs, one for each tree_of_thoughts and graph_of_thoughts.
There is a png file that visualises the graph structure.
There are two json files for storing results from tree_of_thoughts.py and graph_of_thoughts.py.

## How Graph of Thoughts is Implemented

The Graph of Thoughts implementation uses a network-based approach for analyzing relationships between academic papers. At its core, **Graph of Thoughts creates a graph where papers are represented as nodes and their relationships are represented as edges**. This structure enables powerful analysis of how papers relate to each other.

### Key Components:

1. **PaperNode Class**:
   - Extracts content, metadata, authors, citations, and topics from PDFs
   - Methods for parsing text content and different metadata elements
   - Fallback mechanisms for missing metadata (like authors or topics)

2. **GraphOfThoughts Class**:
   - Creates a graph structure where each node is a PaperNode
   - Detects relationships between papers based on:
     - Shared authors
     - Topic overlap
     - Citation overlap
   - Calculates redundancy scores between papers using:
     - LLM-based analysis (using Flan-T5-Small) when available
     - Fallback similarity metrics based on content and metadata overlap
   - Visualization capabilities using NetworkX and Matplotlib
   - JSON export functionality for further analysis

### Innovative Aspects:

- **Redundancy Detection**: Papers are considered redundant if their similarity exceeds a threshold
- **Multi-modal Relationship Analysis**: Captures different types of relationships simultaneously
- **Graceful Degradation**: Works even without LLM access, using simpler heuristics
- **Visual Representation**: Color-codes redundant nodes for easy identification

## How Tree of Thoughts is Implemented

The Tree of Thoughts implementation (in `paste-2.txt`) uses a recursive, tree-structured approach to analyze individual papers in depth. **Tree of Thoughts generates multiple different analytical responses for each paper and then expands the most promising thoughts in subsequent cycles, creating branches of increasingly refined analysis.**

### Key Components:

1. **Thought Class**: 
   - Represents a single analytical insight about a paper
   - Contains a content string and score
   - Can have child thoughts (creating a tree structure)

2. **TreeOfThoughts Class**:
   - Implements a multi-step, branching analysis process
   - Three main components:
     - **Agentic Response**: Generates multiple analytical perspectives using LLM
     - **Critic**: Evaluates and scores each thought based on quality
     - **Decision Maker**: Determines paper publishability based on thought tree
   - Visualization capabilities for the thought tree
   - Support for both LLM-based and simulated analysis

3. **Analysis Process**:
   - Starts with an initial analysis node
   - Generates multiple different responses (typically 3) per cycle
   - Evaluates and scores each response
   - Selects highest-scoring thoughts to expand in the next cycle
   - Each selected thought generates its own set of responses
   - This expansion continues for multiple cycles, creating a branching tree
   - Final decision integrates insights from the entire tree

### Innovative Aspects:

- **Recursive Refinement**: Analysis improves through multiple steps of thought
- **Selection Mechanism**: Only most promising thoughts are developed further
- **Multi-criteria Evaluation**: Papers are assessed on multiple dimensions
- **Visual Decision Trees**: Exports thought progression in visual format

## LLM Integration

Both implementations use the Google Flan-T5-Small model for enhanced analysis:

- **Graph of Thoughts**: Uses LLM for calculating nuanced redundancy scores between papers
- **Tree of Thoughts**: Uses LLM for:
  - Thought generation (simulating expert reviewers)
  - Thought evaluation (quality assessment)
  - Final decision making (publishability verdict)

I've implemented a fallback mechanism that automatically detects when LLM capabilities aren't available and switches to rule-based alternatives.

## Output Formats

Both components generate structured JSON outputs:

1. **Graph of Thoughts**:
   - Paper metadata
   - Relationship details (type, strength)
   - Redundancy assessments
   - Visualization data

2. **Tree of Thoughts**:
   - Complete thought trees with scores
   - Final decisions on publishability
   - Confidence scores
   - Reasoning behind decisions

These outputs enable both human review and potential integration with other systems.
