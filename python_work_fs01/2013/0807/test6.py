#!/usr/bin/env python
import os
import math
f = open("data","w")
for i in xrange(-30,30):
    x=i*0.1
    f.write("%f %f\n" % (x, math.exp(-x*x))
f.close()

command="""gnuplot << EOF
set terminal postscript enhanced
set output "plot.eps"
plot "data" w l
quit
EOF"""

os.system(command)



