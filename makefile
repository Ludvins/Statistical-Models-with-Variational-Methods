main.pdf: main.tex sections/*.tex
	pdflatex --shell-escape main.tex
	bibtex main
	pdflatex --shell-escape main.tex
	pdflatex --shell-escape main.tex
	rm *.aux *.blg *.log *.bbl *.toc *.out
clean:
	rm main.pdf *.aux *.blg *.log *.bbl *.toc *.out
