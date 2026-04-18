MAIN = main

.PHONY: all clean figures

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib icml2026.sty icml2026.bst figures/*.pdf
	pdflatex -interaction=nonstopmode $(MAIN)
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)

figures:
	cd ../../ && uv run python paper/icml/figures/fig1_scaffold.py
	cd ../../ && uv run python paper/icml/figures/fig2_matched.py
	cd ../../ && uv run python paper/icml/figures/fig3_bridge.py

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf $(MAIN).synctex.gz
