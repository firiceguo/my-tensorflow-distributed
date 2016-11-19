def getCPU(file):
    cpu = []
    f = open(file, "r")
    while 1:
        temp = []
        line = f.readline()
        if line:
            if not line[0].isdigit():
                continue
            temp.append(line[:8])
            CpuNum = ''
            Used = ''
            i = 9
            for ch in line[10:]:
                i += 1
                if ch.isdigit():
                    CpuNum += ch
                else:
                    if CpuNum is '':
                        continue
                    else:
                        temp.append(int(CpuNum))
                        break
            for ch in line[i:]:
                if ch.isdigit():
                    Used += ch
                else:
                    if Used is '':
                        continue
                    else:
                        temp.append(int(Used))
                        break
            cpu.append(temp)
        else:
            break
    return(cpu)


def lineCPU(cpu):
    cpu_x, cpu_use = [], []
    for i in xrange(len(cpu)):
        if i % 24 == 0:
            cpu_use.append(cpu[i][2])
            cpu_x.append(i / 24)
        else:
            cpu_use[i / 24] += cpu[i][2]
    return(cpu_x, cpu_use)


def getNet(file):
    '''This func can get the eth0&lo net info.
    Example output: ['10:32:23', 4267, 450]'''
    net = []
    f = open(file, "r")
    while 1:
        temp = []
        line = f.readline()
        if line:
            temp.append(line[:8])
            KBIn = ''
            KBOut = ''
            for ch in line[23:]:
                if ch.isdigit():
                    KBIn += ch
                else:
                    if KBIn is '':
                        continue
                    else:
                        temp.append(int(KBIn))
                        break
            for ch in line[66:]:
                if ch.isdigit():
                    KBOut += ch
                else:
                    if KBOut is '':
                        continue
                    else:
                        temp.append(int(KBOut))
                        break
            net.append(temp)
        else:
            break
    return(net)


def lineNET(eth0, lo):
    eth0_x, eth0_in, eth0_out, lo_x, lo_in, lo_out = [], [], [], [], [], []
    for i in xrange(len(eth0)):
        eth0_x.append(i)
        eth0_in.append(eth0[i][1])
        eth0_out.append(eth0[i][2])
        lo_x.append(i)
        lo_in.append(lo[i][1])
        lo_out.append(lo[i][2])
    return(eth0_x, eth0_in, eth0_out, lo_x, lo_in, lo_out)


def getOut(file):
    acc = []
    f = open(file, "r")
    while 1:
        temp = []
        line = f.readline()
        if line:
            num = []
            accuracy = ''
            for ch in line[13:]:
                if ch.isdigit() or ch is '.':
                    accuracy += ch
                else:
                    if accuracy is '':
                        continue
                    else:
                        temp = float(accuracy)
                        break
            acc.append(temp)
        else:
            break
    return(acc)
