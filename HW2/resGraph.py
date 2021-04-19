from matplotlib import pyplot as plt
models_name = ['AlexNet', 'VGG16', 'ResNet', 'googleNet']
resTop1 = [56.52, 71.59, 69.76, 69.78]
resTop5 = [79.07, 90.38, 89.08, 89.53]

# draw Graph
def create_x(t, w, n, d): # numberOfData, BarWidth, numOfCurrentData, numOfDataLen
    return [t*x + w*n for x in range(d)]
top1_x = create_x(2, 0.8, 1, 4)
top5_x = create_x(2, 0.8, 2, 4)
ax = plt.subplot()
ax.bar(top1_x, resTop1, color='salmon', label='top-1 accuracy')
ax.bar(top5_x, resTop5, color='silver', label='top-5 accuracy')
x = [(a+b)/2 for (a,b) in zip(top1_x, top5_x)]
ax.set_xticks(x)
ax.set_xticklabels(models_name)
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy percentage')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
plt.savefig('modelCompareGraph.png', format='png', dpi=300)
plt.show()