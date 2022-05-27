import pandas
import matplotlib.pyplot as plt

seq_py = pandas.read_csv("results_seq_py.csv")
seq_c = pandas.read_csv("results_seq.csv")
par_c = pandas.read_csv("results_par.csv")


mean_par = par_c.groupby('N').mean()
mean_seq_c = seq_c.groupby('N').mean()
mean_seq_py = seq_py.groupby('N').mean()

print(par_c)
plt.xscale('log')
plt.plot(mean_par['Time'], label="Parallel C")
plt.plot(mean_seq_c['Time'], label="Sequential C")
plt.plot(mean_seq_py['Time'], label="Sequential Python")
plt.scatter(par_c['N'],   par_c['Time'])
plt.scatter(seq_c['N'],   seq_c['Time'])
plt.scatter( seq_py['N'], seq_py['Time'])
plt.xlabel('N')
plt.ylabel('Time(s)')
plt.legend()
plt.show()
