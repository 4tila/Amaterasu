set terminal pdfcairo size 12cm,9cm enhanced font "Times,10"
set output "energy_plots.pdf"

set multiplot layout 3,4 rowsfirst title "Energy Distribution of Each Row"

# General styling
set grid
set key off
set tics font ",8"
set format y "%.2f"
set ytics 1.5
set xtics 10
set style line 1 lc rgb "#2c7fb8" lw 1.8

set yrange [0.0:9.3]

# Loop over 11 channels
set ylabel "Energy"
set label "(A)" at graph 0.1, 0.85
set title sprintf("Row 1") font ",9"
plot "energy.txt" using 0:1 with lines ls 1
unset ylabel
unset label

set label "(B)" at graph 0.1, 0.85
set title sprintf("Row 2") font ",9"
plot "energy.txt" using 0:2 with lines ls 1
unset label

set label "(C)" at graph 0.1, 0.85
set title sprintf("Row 3") font ",9"
plot "energy.txt" using 0:3 with lines ls 1
unset label

set label "(D)" at graph 0.1, 0.85
set ylabel "Energy"
set title sprintf("Row 4") font ",9"
plot "energy.txt" using 0:4 with lines ls 1
unset ylabel
unset label

set label "(E)" at graph 0.1, 0.85
set title sprintf("Row 5") font ",9"
plot "energy.txt" using 0:5 with lines ls 1
unset label

set label "(F)" at graph 0.1, 0.85
set title sprintf("Row 6") font ",9"
plot "energy.txt" using 0:6 with lines ls 1
unset label

set label "(G)" at graph 0.1, 0.85
set title sprintf("Row 7") font ",9"
plot "energy.txt" using 0:7 with lines ls 1
unset label

set label "(H)" at graph 0.1, 0.85
set title sprintf("Row 8") font ",9"
plot "energy.txt" using 0:8 with lines ls 1
unset label

set ylabel "Energy"
set label "(I)" at graph 0.1, 0.85
set title sprintf("Row 9") font ",9"
plot "energy.txt" using 0:9 with lines ls 1
unset ylabel
unset label

set label "(J)" at graph 0.1, 0.85
set title sprintf("Row 10") font ",9"
plot "energy.txt" using 0:10 with lines ls 1
unset label

set label "(K)" at graph 0.1, 0.85
set title sprintf("Row 11") font ",9"
plot "energy.txt" using 0:11 with lines ls 1
unset label

# Optional: empty 12th plot (for spacing)
unset title
unset multiplot

