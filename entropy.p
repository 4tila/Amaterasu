set terminal pdfcairo enhanced font "Times,12" size 8cm,6cm
set output "entropy_plot.pdf"

set title "Entropy Decay"
unset xlabel
set ylabel "Entropy"

set grid lw 0.5
set border lw 1.2

set xtics out
set ytics 0.4

set format y "%.1f"

set key off

plot "entropy.txt" using 0:1 with lines lw 2 title "Entropy"

