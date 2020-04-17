main.pdf: main.tex sections/*.tex
	pdflatex --shell-escape main.tex -interaction=nonstopmode
	bibtex main
	pdflatex --shell-escape main.tex -interaction=nonstopmode
	pdflatex --shell-escape main.tex -interaction=nonstopmode
clean:
	rm main.pdf *.aux *.blg *.log *.bbl *.toc *.out
