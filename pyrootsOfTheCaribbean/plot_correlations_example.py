import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot1D(inputdf, var="E1", range_=(300, 900)):
    plt.figure()
    plt.hist(inputdf[var].loc[inputdf["signal"] == 1], bins=40,
             edgecolor="r", range=range_, histtype="step", label="signal")
    plt.hist(inputdf[var].loc[inputdf["signal"] == 0], bins=40,
             edgecolor="b", range=range_, histtype="step", label="bkg")
    plt.xlabel(var)
    plt.ylabel("Events")
    plt.legend(loc='upper left')
    plt.savefig(var+".pdf")

    plt.show()
    plt.draw()
    plt.close()


def plot2D(inputdf, vars=["E1", "E2"], bins_=[50, 50], xrange=(300, 900), yrange=(300, 900)):
    plt.figure()
    plt.hist2d(  x=inputdf[vars[0]].loc[inputdf["signal"] == 1],
                 y=inputdf[vars[1]].loc[inputdf["signal"] == 1], bins=bins_,
                 range=[ [ xrange[0], xrange[1] ], [ yrange[0], yrange[1] ] ] 
                )
    plt.savefig(vars[0]+vars[1]+".pdf")

    plt.show()
    plt.draw()
    plt.close()

def plotScatter(inputdf, vars=["E1", "E2"], xrange=(300, 900), yrange=(300, 900)):
    plt.figure()
    plt.scatter( x=inputdf[vars[0]].loc[inputdf["signal"] == 1],
                 y=inputdf[vars[1]].loc[inputdf["signal"] == 1],
                #  range=[ [ xrange[0], xrange[1] ], [ yrange[0], yrange[1] ] ], 
                c = "b",
                label="bkg"
                )

    plt.scatter( x=inputdf[vars[0]].loc[inputdf["signal"] == 0],
                 y=inputdf[vars[1]].loc[inputdf["signal"] == 0],
                 c="r",
                #  range=[ [ xrange[0], xrange[1] ], [ yrange[0], yrange[1] ] ], 
                label="signal"
                )
    plt.legend(loc='upper left')
    plt.xlabel(vars[0])
    plt.ylabel(vars[1])
    plt.savefig(vars[0]+vars[1]+".pdf")

    plt.show()
    plt.draw()
    plt.close()


def plotScatter3D(inputdf, vars=["E1", "E2", "dPhi"], xrange=(300, 900), yrange=(300, 900), zrange=(0,3.2)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter( inputdf[vars[0]].loc[inputdf["signal"] == 1],
                inputdf[vars[1]].loc[inputdf["signal"] == 1],
                inputdf[vars[2]].loc[inputdf["signal"] == 1],
                #  range=[ [ xrange[0], xrange[1] ], [ yrange[0], yrange[1] ] ] 
                c = "b",
                label="bkg"
                )

    ax.scatter( inputdf[vars[0]].loc[inputdf["signal"] == 0],
                inputdf[vars[1]].loc[inputdf["signal"] == 0],
                inputdf[vars[2]].loc[inputdf["signal"] == 0],
                c="r",
                #  range=[ [ xrange[0], xrange[1] ], [ yrange[0], yrange[1] ] ] 
                label="signal"
                )
    # plt.legend(loc='upper left')
    ax.set_xlabel(vars[0])
    ax.set_ylabel(vars[1])
    ax.set_zlabel(vars[2])
    
    plt.savefig(vars[0]+vars[1]+vars[2]+".pdf")
    plt.show()
    plt.draw()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter( inputdf[vars[0]].loc[inputdf["signal"] == 1],
        inputdf[vars[1]].loc[inputdf["signal"] == 1],
        inputdf[vars[2]].loc[inputdf["signal"] == 1],
        #  range=[ [ xrange[0], xrange[1] ], [ yrange[0], yrange[1] ] ] 
        c = "b",
        label="bkg"
        )

    ax.set_xlabel(vars[0])
    ax.set_ylabel(vars[1])
    ax.set_zlabel(vars[2])

    plt.savefig(vars[0]+vars[1]+vars[2]+"_signal.pdf")
    plt.show()
    plt.draw()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter( inputdf[vars[0]].loc[inputdf["signal"] == 0],
                inputdf[vars[1]].loc[inputdf["signal"] == 0],
                inputdf[vars[2]].loc[inputdf["signal"] == 0],
                c="r",
                #  range=[ [ xrange[0], xrange[1] ], [ yrange[0], yrange[1] ] ] 
                label="signal"
                )

    ax.set_xlabel(vars[0])
    ax.set_ylabel(vars[1])
    ax.set_zlabel(vars[2])

    plt.savefig(vars[0]+vars[1]+vars[2]+"_bkg.pdf")
    plt.show()
    plt.draw()
    plt.close()
    





if __name__ == "__main__":
    dfPath= "/ceph/swieland/TP2-WZH/MLInput/train/data.h5"
    df = pd.read_hdf(dfPath)
    print(df.loc[df["signal"]==1])
    print(df.loc[df["signal"]==0])
    plot1D(inputdf=df, var="E1")
    plot1D(inputdf=df, var="E2")
    plot1D(inputdf=df, var="pt1")
    plot1D(inputdf=df, var="pt2")
    plot1D(inputdf=df, var="dPhi", range_=(0, 3.2))

    # plotScatter(inputdf=df, vars=["E1", "E2"])
    # plotScatter(inputdf=df, vars=["E1", "dPhi"])
    # plotScatter(inputdf=df, vars=["E2", "dPhi"])
    # plotScatter3D(inputdf=df, vars=["E1", "E2", "dPhi"])
