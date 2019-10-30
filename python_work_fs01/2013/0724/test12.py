#!/usr/bin/env python
atom=[""]
tau_x=[0]
tau_y=[0]
tau_z=[0]
c=0.529 # BohrtoAA 
f=open("sample/sample3.out","r")
for line in f:
    if "lattice" in line:
        data = line.split()
        a = float(data[4])
    if "tau" in line:
        data = line.split()
        atom += [data[1]]
        tau_x += [float(data[6])*a*c]
        tau_y += [float(data[7])*a*c]
        tau_z += [float(data[8])*a*c]
f.close()

f=open("sample3.xyz","w")
f.write("%d\n" % (len(tau_x)-1) )
f.write("Structure\n")
for i in xrange(1,len(tau_x)):
    f.write("%s %f %f %f\n" % (atom[i], tau_x[i], tau_y[i], tau_z[i]))  #  -> H
    # f.write("%r %f %f %f\n" % (atom[i], tau_x[i], tau_y[i], tau_z[i])) -> 'H'
    # f.write("atom[i], tau_x[i], tau_y[i], tau_z[i]")                   ->  atom[i]
    print atom[i], tau_x[i], tau_y[i], tau_z[i]

