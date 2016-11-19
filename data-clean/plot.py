from utils import *
import matplotlib.pyplot as plt

num = "1"
logdir = "../log-m2d6/3/"
cpu = getCPU(logdir + "CPU" + num)
eth0 = getNet(logdir + "Net" + num + "-eth0")
lo = getNet(logdir + "Net" + num + "-lo")
out = getOut(logdir + "output-" + num + "-acc")

eth0_x = []
eth0_in = []
eth0_out = []
lo_x = []
lo_in = []
lo_out = []
cpu_x = []
cpu_use = []

eth0_x, eth0_in, eth0_out, lo_x, lo_in, lo_out = lineNET(eth0, lo)
cpu_x, cpu_use = lineCPU(cpu)

plt.plot(cpu_x, cpu_use)
plt.savefig(logdir + "cpu_use_" + num + ".png")
plt.close('all')

plt.plot(eth0_x, eth0_out, 'g')
plt.plot(eth0_x, eth0_in, 'r')
plt.savefig(logdir + "eth0_" + num + ".png")
plt.close('all')

plt.plot(lo_x, lo_in, 'g')
plt.plot(lo_x, lo_out, 'r')
plt.savefig(logdir + "lo_" + num + ".png")
plt.close('all')

plt.plot(out)
plt.savefig(logdir + "output_" + num + ".png")
plt.close('all')

# plt.show()
