# LMParsCit

This project leverages LLMs (e.g., llama3-8b-instruct) to extract key metadata fields—like title, author, venue, and year—from references across a range of bibliography types, including journals, conference proceedings, book chapters, tech reports, and theses. It also supports multiple citation styles (e.g., IEEE, ACM, APA) and spans different academic disciplines, including STEM and non-STEM majors.

## Performed Instruct-tuning LLM fine-tuning process:
**Dataset**
-- dataset/GIANT-training_data: we used 1K subset and 10k subset for training, both porivded similar performance
-- dataset/EvaluationBenchmark: consists of raw annotated and processed citation strings of two dataset for evaluation: ETDCite (**Ours**) and CORA

**Code**
-- misc-experiment/NeuralParsCit: used the script to test on ETDCite and achieved its performance on ETDCite
-- src/:
	-- citation-preprocessing.py: takes subset of GITANT dataset, convert into chat template for instruction tuning
	-- citation-finetuning.py: leverages a 4-bit quantized llama-3-8b-instruct model, and further uses LoRA adapter in fine-tuning process. It saves the model and tokenizer.
	-- citation-evaluation.py: It loads the model and tokenizer to perform evaluation on ETDCite and CORA.


