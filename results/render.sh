mkdir -p ../docs
quarto render results.qmd
cp results.html ../docs/results.html

# if you want to render to markdown (e.g. for a blog elsewhere)
# quarto render results.qmd --to markdown --output results.md