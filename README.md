Neural Machine Translation with Transformer
Custom implementation of the Transformer architecture for English to German translation, based on "Attention is All You Need" paper.

Project Overview
A transformer-based neural machine translation system implemented from scratch using PyTorch, capable of translating English text to German.

Features
Custom Transformer implementation from scratch
Multi-head attention mechanism
Positional encoding
Encoder-decoder architecture
Interactive translation interface
Directory Structure
transformer_from_scratch/ ├── src/ │ ├── model.py # Transformer implementation │ └── translate.py # Translation interface ├── tokenizers/ │ ├── spm_de.model # German tokenizer │ └── spm_en.model # English tokenizer └── data/ └── raw/ # Training and test data

Installation
Clone the repository
Install requirements:
pip install -r requirements.txt

Usage
Run the translation interface:

python src/translate.py

Example
Input: "Hello, how are you?" Output: "Hallo, wie geht es dir?"

Implementation Details
Transformer architecture with multi-head attention
Custom positional encoding
SentencePiece tokenization
PyTorch implementation
References
Attention Is All You Need
