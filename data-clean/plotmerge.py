from utils import *
import matplotlib.pyplot as plt

logdir = "../log-m2d6/3/"
num0 = "0"
# cpu_0 = getCPU(logdir + "CPU" + num0)
eth0_0 = getNet(logdir + "Net" + num0 + "-eth0")
lo_0 = getNet(logdir + "Net" + num0 + "-lo")
# out0 = getOut(logdir + "output-" + num0 + "-acc")

num1 = "1"
# cpu_1 = getCPU(logdir + "CPU" + num1)
eth0_1 = getNet(logdir + "Net" + num1 + "-eth0")
lo_1 = getNet(logdir + "Net" + num1 + "-lo")
# out1 = getOut(logdir + "output-" + num1 + "-acc")

eth0_0_x = []
eth0_0_in = []
eth0_0_out = []
lo_0_x = []
lo_0_in = []
lo_0_out = []

eth0_1_x = []
eth0_1_in = []
eth0_1_out = []
lo_1_x = []
lo_1_in = []
lo_1_out = []
# cpu_0_x = []
# cpu_1_x = []
# cpu_0_use = []
# cpu_1_use = []

eth0_0_x, eth0_0_in, eth0_0_out, lo_0_x, lo_0_in, lo_0_out = lineNET(eth0_0, lo_0)
eth0_1_x, eth0_1_in, eth0_1_out, lo_1_x, lo_1_in, lo_1_out = lineNET(eth0_1, lo_1)
# cpu_0_x, cpu_0_use = lineCPU(cpu_0)
# cpu_1_x, cpu_1_use = lineCPU(cpu_1)

# plt.plot(cpu_0_x, cpu_0_use, "r")
# plt.plot(cpu_1_x, cpu_1_use, "g")
# plt.show()

# plt.plot(out0, "r")
# plt.plot(out1, "g")
# plt.show()

plt.plot(eth0_0_x, eth0_0_out, 'g')
plt.plot(eth0_1_x, eth0_1_out, 'r')
plt.show()
