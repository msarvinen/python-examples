import matplotlib.pyplot as plt
import numpy as np
import time

class DynamicUpdate():
    plt.ion()
    #Suppose we know the x range
    min_x = 0
    max_x = 2*3.1415926536
    min_y = -1.5
    max_y = 1.5

    def on_launch(self):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines1, = self.ax.plot([],[])
        self.lines2, = self.ax.plot([],[])
        plt.xlabel('radians')
        plt.ylabel('amplitude')
        plt.title('sin and cos cycle')
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.axis(ymin=self.min_y, ymax=self.max_y)
        #Other stuff
        self.ax.grid()

    def on_running(self, xdata, ydata1, ydata2):
        #Update data (with the new _and_ the old points)
        self.lines1.set_xdata(xdata)
        self.lines1.set_ydata(ydata1)
        self.lines2.set_xdata(xdata)
        self.lines2.set_ydata(ydata2)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    #Example
    def __call__(self):
        return
        
if __name__ == "__main__":
    pi = 3.1415926536
    d = DynamicUpdate()
    d.on_launch()
    xdata = []
    ydata1 = []
    ydata2 = []

    for x in np.arange(0, 2 * pi, pi/90):
        xdata.append(x)
        ydata1.append((1*np.sin(x)))
        ydata2.append((1*np.cos(x)))
        d.on_running(xdata, ydata1, ydata2)
        time.sleep(0.001)
        

    input("press any key to continue")