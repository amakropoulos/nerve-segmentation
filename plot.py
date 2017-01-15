import argparse
from matplotlib import pyplot
import misc

def plot_errors(version=0):
    mve, train_error, val_error, val_accuracy = misc.load_results(version)
    train_loss = []
    for value in train_error.values(): train_loss.append(abs(float(value)))
    valid_loss = []
    for value in val_error.values(): valid_loss.append(abs(float(value)))
    valid_ratio = np.divide(valid_loss,train_loss)
    
    pyplot.subplots(1, 2, sharey=True)
    ax = pyplot.subplot(2, 1, 1)
    ax.set_ylim((0, 1))
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend(loc=4)
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")

    pyplot.subplot(2, 1, 2)
    pyplot.plot(valid_ratio, linewidth=3, label="valid")
    pyplot.show()



def main(): 
    parser.add_argument("-v", "--version", dest="version",  help="version", default=0.0)   
    plot_errors(version)


if __name__ == '__main__':
    main()

