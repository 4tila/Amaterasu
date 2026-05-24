set terminal pngcairo size 1200,900 enhanced font 'Arial,10'
set output 'singular_curvature_comparison.png'

# General styling
set grid
set key off

# Multiplot layout: 2 rows, 2 columns
set multiplot layout 2,2 title "Deterministic vs Stochastic System Analysis"

set yrange [0:350]
set xrange [0:300]
# -------------------------------
# Top Left: Deterministic Singular Values
# -------------------------------
set title "Deterministic - Singular Values"
set label "(A)" at graph 0.1, 0.95
unset xlabel
#et xlabel "Index"
set ylabel "Singular Value"
plot "/media/atila/C117-51DB/books/deterministic_S.txt" using 1 with lines lw 2
unset label

# -------------------------------
# Top Right: Stochastic Singular Values
# -------------------------------
set title "Stochastic - Singular Values"
set label "(B)" at graph 0.1, 0.95
unset xlabel
unset ylabel
#set xlabel "Index"
#set ylabel "Singular Value"
plot "/media/atila/C117-51DB/books/stochastic_S.txt" using 1 with lines lw 2
unset label

set yrange [0:0.8]
# -------------------------------
# Bottom Left: Deterministic Curvature
# -------------------------------
set title "Deterministic - Curvature (2nd Derivative)"
set label "(C)" at graph 0.1, 0.95
set xlabel "Index"
set ylabel "Curvature"
plot "/media/atila/C117-51DB/books/deterministic_S.txt" using 2 with lines lw 2
unset label

# -------------------------------
# Bottom Right: Stochastic Curvature
# -------------------------------
set title "Stochastic - Curvature (2nd Derivative)"
set label "(D)" at graph 0.1, 0.95
set xlabel "Index"
unset ylabel
#set ylabel "Curvature"
plot "/media/atila/C117-51DB/books/stochastic_S.txt" using 2 with lines lw 2
unset label

unset multiplot

