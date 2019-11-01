#!/usr/bin/env python
import os
print os.system("ls")
import commands
results = commands.getoutput("ls")
print results

