MAIN = main
STYLE_DIR = guidelines-style-files
TEXINPUTS_BUILD = $(STYLE_DIR):
BSTINPUTS_BUILD = $(STYLE_DIR):

.PHONY: all clean figures

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib $(STYLE_DIR)/icml2026.sty $(STYLE_DIR)/icml2026.bst figures/*.pdf
	TEXINPUTS=$(TEXINPUTS_BUILD) pdflatex -interaction=nonstopmode $(MAIN)
	BSTINPUTS=$(BSTINPUTS_BUILD) bibtex $(MAIN)
	TEXINPUTS=$(TEXINPUTS_BUILD) pdflatex -interaction=nonstopmode $(MAIN)
	TEXINPUTS=$(TEXINPUTS_BUILD) pdflatex -interaction=nonstopmode $(MAIN)

figures:
	cd ../../ && uv run python paper/icml/figures/fig1_scaffold.py
	cd ../../ && uv run python paper/icml/figures/fig2_matched.py
	cd ../../ && uv run python paper/icml/figures/fig3_bridge.py

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf $(MAIN).synctex.gz
