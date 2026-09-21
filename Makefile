.PHONY: help install test demo app pull-model clean

# Mặc định: hiện help
help:
	@echo "Scientific Debate Simulator — Make targets:"
	@echo ""
	@echo "  make install      Cài project (pip install -e .)"
	@echo "  make pull-model   Pull qwen2.5:3b qua Ollama"
	@echo "  make test         Chạy tất cả unit tests"
	@echo "  make demo         Chạy CLI demo với claim mặc định"
	@echo "  make app          Chạy Streamlit UI"
	@echo "  make clean        Xóa cache + __pycache__"
	@echo ""

install:
	pip install --upgrade pip
	pip install -e .

pull-model:
	ollama pull qwen2.5:3b

test:
	@echo "Running test_llm..."
	python tests/test_llm.py
	@echo ""
	@echo "Running test_memory..."
	python tests/test_memory.py
	@echo ""
	@echo "Running test_prompts..."
	python tests/test_prompts.py
	@echo ""
	@echo "All tests passed."

demo:
	python experiments/run_demo.py --rounds 1

app:
	streamlit run app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache *.egg-info build dist
	@echo "Cleaned cache files."