.PHONY: main.pdf clean


main.pdf: main.tex tex/*/*.tex
	pdflatex --shell-escape main.tex
	bibtex main
	pdflatex --shell-escape main.tex
	pdflatex --shell-escape main.tex
clean:
	rm main.pdf *.aux *.blg *.log *.bbl *.toc *.out
