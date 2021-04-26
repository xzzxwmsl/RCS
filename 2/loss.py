import csv
import matplotlib
import matplotlib.pyplot as plt

D_step = []
D_loss = []
G_step = []
G_loss = []

D_loss_path = 'D_losses.csv'
G_loss_path = 'G_losses.csv'
with open(D_loss_path) as f:
    reader = csv.reader(f)
    for row in reader:
        D_step.append(float(row[0]))
        D_loss.append(float(row[1]))

with open(G_loss_path) as f:
    reader = csv.reader(f)
    for row in reader:
        G_step.append(float(row[0]))
        G_loss.append(float(row[1]))
plt.plot(G_step,G_loss,label='G_loss')
plt.plot(D_step,D_loss,label='D_loss')
plt.title('LOSS')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.legend()
plt.show()
