mkdir -p ../docs
quarto render results.qmd
cp results.html ../docs/results.html
