#!/usr/bin/env python
import os

command="""gnuplot << EOF
plot sin(x) title "sin function"
set terminal postscript enhanced
set output "sample.eps"
replot
quit
EOF"""

os.system(command)



