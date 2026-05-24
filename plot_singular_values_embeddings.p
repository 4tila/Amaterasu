set terminal pngcairo size 1000,700 enhanced font 'Arial,12'
set output 'singular_values_plot.png'

set title "Singular Value Spectrum (spaCy embeddings)"
set xlabel "Component Index"
set ylabel "Singular Value"

set grid

# Optional: log scale (very useful for spectra)
# uncomment if needed
# set logscale y

# Vertical line at x = 58
set arrow from 58, graph 0 to 58, graph 1 nohead lc rgb "red" lw 2 dt 2

# Label for the cutoff
set label "Cutoff (58)" at 60, graph 0.9 tc rgb "red"

plot "/media/atila/C117-51DB/books/vector_embedding_singular_value.txt" \
     using 0:1 with lines lw 2 title "Singular Values"

