# Causal Tracing in Large Language Models (LLMs)

This project explores causal tracing techniques in large language models, focusing on understanding how internal representations contribute to model predictions. Causal tracing involves systematically intervening in the model's activations to identify which components are responsible for specific behaviors, such as factual recall, logical reasoning, and decision-making in yes/no questions.

## Project Overview

The project is divided into two main parts, implemented in Jupyter notebooks:

### Part 1: Causal Tracing Experiments (`notebook_part1.ipynb`)

This part investigates causal tracing on factual knowledge and boolean question-answering tasks using GPT-2 XL models.

- **Factual Knowledge Tracing**: We perform causal tracing on prompts related to world knowledge, such as "The capital of Germany is" or "The Mona Lisa was painted by". By corrupting subject tokens and restoring activations at different layers and positions, we identify where factual information is stored and retrieved in the model.

- **Boolean Question Answering (BoolQ)**: Using the BoolQ dataset, we compare causal tracing results between a base GPT-2 XL model and a fine-tuned BoolQ model. This highlights how fine-tuning affects the storage and retrieval of task-specific knowledge, particularly for yes/no questions.

Key techniques include:
- Noise injection for corruption.
- Restoration of clean activations in intervals (single layer for block output, 10-layer windows for MLP and attention outputs).
- Visualization of probability differences across layers, positions, and streams (block output, MLP activation, attention output).

### Part 2: Probing Activations (`notebook_part2.ipynb`)

This part focuses on probing the model's internal activations to understand how it handles logical reasoning tasks, specifically room recognition and color induction from the bAbI dataset.

- **Room Recognition**: Using the "agents-motivations" task, we train linear probes on activations at different layers and token positions to predict which room an agent will go to based on their current state (e.g., tired, hungry). This reveals how the model encodes state information and makes predictions.

- **Color Induction**: Using the "basic-induction" task, we probe for color predictions based on inductive reasoning (e.g., associating colors with animal categories). Probes are trained on positions corresponding to color sources, animals, subjects, and the last token.

Key techniques include:
- Extracting activations from specific token positions.
- Training logistic regression probes for classification.
- Heatmap visualizations of probe accuracy across layers and positions.

## Technologies and Libraries

- **Models**: GPT-2 XL (base and BoolQ fine-tuned).
- **Libraries**: PyTorch, Transformers, Datasets, PyVene (for interventions), Scikit-learn, Matplotlib, Seaborn, Plotnine.
- **Datasets**: bAbI (for probing), BoolQ (for causal tracing).

## Key Findings

- Causal tracing reveals that factual knowledge is often stored in middle-to-late layers, with attention outputs playing a crucial role in retrieval.
- Probing shows that state and inductive information is encoded in specific positions, with higher accuracy in later layers.
- Fine-tuning on BoolQ shifts the causal traces, indicating changes in how knowledge is represented.

## How to Run

1. Install dependencies: `pip install torch transformers datasets pyvene scikit-learn matplotlib seaborn plotnine`.
2. Open the notebooks in VS Code or Jupyter.
3. Run cells sequentially; some computations may take time due to model size.

## Academic Context

This project was developed as "Lista 5" (Assignment 5), Natural Language Processing course, showcasing practical implementation of state-of-the-art RAG techniques.