import pickle
import matplotlib.pyplot as plt
import jax.numpy as np
time_list = pickle.load( open( "data/full_simulation_comparison2.p", "rb" ) )
Mp_list = time_list[-2]
Ms_list = time_list[-1]
Ntot_list = []

linsolve_reorder_avg,tot_reorder_avg,eval_reorder_avg,init_reorder_avg,overhead_reorder_avg, _, _ = pickle.load( open( "data/reorder_simulation.p", "rb" ) )
for i in range(0, len(Mp_list)):
    Ntot_pe = (Mp_list[i] + 2) * (Mp_list[i]) + 4 * (Mp_list[i] + 2) + 2 * (Mp_list[i])
    Ntot_ne = (Mp_list[i] + 2) * (Mp_list[i]) + 4 * (Mp_list[i] + 2) + 2 * (Mp_list[i])

    Ntot_sep = 3 * (Ms_list[i] + 2)
    Ntot_acc = Ms_list[i] + 2
    Ntot_zcc = Ms_list[i] + 2
    Ntot = Ntot_pe + Ntot_ne + Ntot_sep + Ntot_acc + Ntot_zcc
    Ntot_list.append(Ntot)

linsolve_list = time_list[0:3]
eval_list = time_list[6:9]
tot_list = time_list[3:6]
overhead_list = time_list[12:15]
init_list = time_list[9:12]

plt.figure()
ax = plt.gca()
plt.rcParams.update({'font.size': 16})
plt.semilogy(Ntot_list,tot_list[0],'k+-',label="naive");
# plt.semilogy(Ntot_list,(tot_list[1]),'r+-',label="decoupled");
plt.semilogy(Ntot_list,tot_reorder_avg,'r+-',label="fast");
ax.grid(which='major', axis='both')
plt.xlabel("Degree of freedom")
plt.ylabel("Total simulation time (s)")
plt.legend(loc='best');
plt.savefig('figs/full_total_5to30.pdf', bbox_inches='tight')

plt.figure()
ax = plt.gca()
plt.rcParams.update({'font.size': 16})
plt.plot(Ntot_list,linsolve_list[0],'k+-',label="naive");
# plt.semilogy(Ntot_list,(linsolve_list[1]),'r+-',label="decoupled");
plt.plot(Ntot_list,linsolve_reorder_avg,'r+-',label="fast");
ax.grid(which='major', axis='both')
plt.xlabel("Degree of freedom")
plt.ylabel("Linear solve time (s)")
plt.legend(loc='best');
plt.savefig('figs/full_linsolve_5to30.pdf', bbox_inches='tight')

plt.figure()
ax = plt.gca()
plt.rcParams.update({'font.size': 16})
plt.semilogy(Ntot_list[0:3],(eval_list[0][0][0:3]),'k+-',label="naive");
plt.semilogy(Ntot_list[0:3],(eval_list[1][0][0:3]),'r+-',label="decoupled");
plt.semilogy(Ntot_list[0:3],(eval_list[2][0][0:3]),'b+-',label="reordered");
ax.grid(which='major', axis='both')
plt.xlabel("Degree of freedom")
plt.ylabel("Function evaluation time")
plt.legend(loc='best');
plt.savefig('full_eval.pdf', bbox_inches='tight')