.PHONY: main.pdf clean


main.pdf: Text/main.tex Chapters/*/*.tex
	pdflatex --shell-escape Text/main.tex
	bibtex main
	pdflatex --shell-escape Text/main.tex
	pdflatex --shell-escape Text/main.tex
clean:
	rm main.pdf *.aux *.blg *.log *.bbl *.toc *.out
