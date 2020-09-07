.PHONY: main.pdf clean

slideshow.pdf: slideshow.tex
	pdflatex --shell-escape slideshow.tex

main.pdf: main.tex tex/*/*.tex
	pdflatex --shell-escape main.tex
	bibtex main
	makeglossaries main
	pdflatex --shell-escape main.tex
	pdflatex --shell-escape main.tex
clean:
	rm main.aux main.blg main.gls main.nlo main.run.xml main.bbl main.glg main.ist main.out main.sbl main.toc main.bcf main.glo main.log main.sym slideshow.aux slideshow.log slideshow.nav slideshow.out slideshow.snm slideshow.toc
