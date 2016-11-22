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
plt.xlabel('Time (s)')
plt.ylabel('Percentage')
plt.title('CPU Utilization')
plt.grid(True)
plt.savefig(logdir + "cpu_use_" + num + ".png")
plt.close('all')

plt.plot(eth0_x, eth0_out, 'g', label="Outbound")
plt.plot(eth0_x, eth0_in, 'r', label="Inbound")
plt.xlabel('Time (s)')
plt.ylabel('Percentage')
plt.title('Network Utilization')
plt.grid(True)
plt.legend()
plt.savefig(logdir + "eth0_" + num + ".png")
plt.close('all')

plt.plot(lo_x, lo_in, 'g', label="Outbound")
plt.plot(lo_x, lo_out, 'r', label="Inbound")
plt.xlabel('Time (s)')
plt.ylabel('Percentage')
plt.title('Network Utilization (Local Loopback)')
plt.grid(True)
plt.legend()
plt.savefig(logdir + "lo_" + num + ".png")
plt.close('all')

plt.plot(out)
plt.xlabel('')
plt.ylabel('')
plt.title('Output')
plt.grid(True)
plt.savefig(logdir + "output_" + num + ".png")
plt.close('all')

# plt.show()
