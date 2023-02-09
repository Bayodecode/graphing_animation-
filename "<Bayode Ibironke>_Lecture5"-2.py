#!/usr/bin/env python
# coding: utf-8

# ## Lecture 5: Graphics and Animation, Part 1 + Markdown/Latex 

# In[3]:


from pylab import plot,show
y = [ 1.0, 2.4, 1.7, 0.3, 0.6, 1.8]
plot (y)
show()


# In[4]:


from pylab import plot,show
x = [ 0.5, 1.0, 2.0, 4.0, 7.0, 10.0 ] 
y = [ 1.0, 2.4, 1.7, 0.3, 0.6, 1.8] 
plot(x,y)
show()


# In[5]:


from pylab import plot, show 
from numpy import linspace, sin
x =linspace(0,10,100) 
y = sin(x)
plot(x,y)
show()


# In[15]:


from pylab import plot, show 
from numpy import linspace, sin, cos
x =linspace(0,10,100) 
y = cos(x)
plot(x,y)
show()


# In[14]:


from pylab import plot,show 
from math import sin
from numpy import linspace

xpoints = []
ypoints = []
for x in linspace(0,10,100):
    xpoints.append(x) 
    ypoints.append(sin(x))

plot(xpoints,ypoints) 
show()


# In[24]:


from pylab import plot,ylim,show, xlabel, ylabel
from numpy import linspace,sin

x = linspace(0,10,100) 
y = sin(x)

plot(x,y)
ylim(-1.1, 1.1)
xlabel("x axis") 
ylabel("y = sin x") 
show()


# ### Graph styling 

# In[25]:


from pylab import plot,ylim,show, xlabel, ylabel
from numpy import linspace,sin

x = linspace(0,10,100) 
y = sin(x)

plot(x,y, "g--")
ylim(-1.1, 1.1)
xlabel("x axis") 
ylabel("y = sin x") 
show()


# 

# In[27]:


from pylab import plot,ylim,show, xlabel, ylabel
from numpy import linspace,sin

x = linspace(0,10,100) 
y = sin(x)

plot(x,y, "ks")
ylim(-1.1, 1.1)
xlabel("x axis") 
ylabel("y = sin x") 
show()


# In[29]:


from pylab import plot,ylim,xlabel,ylabel,show 
from numpy import linspace,sin,cos

x =linspace(0,10,100) 
y1 = sin(x)
y2 =cos(x) 
plot(x,y1,"k-")
plot (x,y2, "b--")
ylim(-1.1, 1.1)
xlabel("x axis")
ylabel("y =sin x or y cos x") 
show()


# ### Exercise 3.1: Plotting experimental data
# 
# In the on-line resources3 you will find a file called sunspots .txt, which contains the observed number of sunspots on the Sun for each month since January 1749. The file contains two columns of numbers, the first being the month and the second being the sunspot number.
# 
# a) Write a program that reads in the data and makes a graph of sunspots as a function of time.
# b) Modify your program to display only the first 1000 data points on the graph.
# c) Modify your program further to calculate and plot the running average of the data, defined by
# 
# 
# Y_k = 1/(2r + 1) * ∑_{m=-r}^{r} y_{k + m} 
# 
# where r = 5 in this case (and the Y_k are the sunspot numbers). Have the program plot both the original data and the running average on the same graph, again over the range covered by the first 1000 data point

# In[2]:


with open("/Users/bayodeibironke/Downloads/cpresources/sunspots.txt") as file:
    data = file.readlines()
    
# Do something with the data
for line in data:
    print(line)


# In[4]:


import matplotlib.pyplot as plt

# Read the data into the script
time = []
sunspots = []
with open("/Users/bayodeibironke/Downloads/cpresources/sunspots.txt") as file:
    for line in file:
        # Split the line into two parts: time and sunspot count
        parts = line.split()
        time.append(float(parts[0]))
        sunspots.append(float(parts[1]))
        
# Make the graph
plt.plot(time, sunspots)
plt.xlabel("Time (in months)")
plt.ylabel("Sunspot count")
plt.title("Sunspots as a function of time")
plt.show()


# In[9]:


import matplotlib.pyplot as plt

# Read the data into the script
time = []
sunspots = []
with open("/Users/bayodeibironke/Downloads/cpresources/sunspots.txt") as file:
    for line in file:
        # Split the line into two parts: time and sunspot count
        parts = line.split()
        time.append(float(parts[0]))
        sunspots.append(float(parts[1]))
        
# Only display the first 1000 data points
time = time[:1000]
sunspots = sunspots[:1000]

# Make the graph
plt.plot(time, sunspots)
plt.xlabel("Time (in months)")
plt.ylabel("Sunspot count")
plt.title("Sunspots as a function of time (first 1000 data points)")
plt.show()


# In[10]:


import matplotlib.pyplot as plt

# Read the data into the script
time = []
sunspots = []
with open("/Users/bayodeibironke/Downloads/cpresources/sunspots.txt") as file:
    for line in file:
        # Split the line into two parts: time and sunspot count
        parts = line.split()
        time.append(float(parts[0]))
        sunspots.append(float(parts[1]))
        
# Truncate the data to the first 1000 data points
time = time[:1000]
sunspots = sunspots[:1000]

# Calculate the running average of the data
r = 5
running_average = []
for i in range(len(sunspots)):
    start = max(0, i - r)
    end = min(len(sunspots), i + r + 1)
    window = sunspots[start:end]
    average = sum(window) / len(window)
    running_average.append(average)

# Make the graph
plt.plot(time, sunspots, label="Original data")
plt.plot(time, running_average, label="Running average")
plt.xlabel("Time (in months)")
plt.ylabel("Sunspot count")
plt.title("Sunspots and Running Average as a Function of Time")
plt.legend()
plt.show()


# ### Exercise 3.2: Curve plotting
# 
# 
# Although the plot function is designed primarily for plotting standard xy graphs, it can be adapted for other kinds of plotting as well.
# 
# a) Make a plot of the so-called deltoid curve, which is defined parametrically by the equations
# 
# x =2cos theta +cos2(theta) and y =2sintheta - sin2theta,
# where 0 </- theta < 2 pi. Take a set of values of theta between zero and 2pi and calculate x and y for each from the equations above, then plot y as a function of x.

# In[11]:


import numpy as np
import matplotlib.pyplot as plt

# Create an array of theta values from 0 to 2*pi with 1000 steps
theta = np.linspace(0, 2*np.pi, 1000)

# Calculate x and y using the equations
x = 2 * np.cos(theta) + np.cos(2 * theta)
y = 2 * np.sin(theta) - np.sin(2 * theta)

# Plot y vs x
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Deltoid curve')
plt.show()


# b) Taking this approach a step further, one can make a polar plot r = f(theta) for some function f by calculating r for a range of values of theta and then converting r and theta to Cartesian coordinates using the standard equations x = rcos(theta), y = rsin(theta). Use this method to make a plot of the Galilean spiral 
# r = theta ^ 2 for 0 </ = theta </= 10 pi.

# In[12]:


import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0, 10 * np.pi, 1000)
r = theta ** 2
x = r * np.cos(theta)
y = r * np.sin(theta)

plt.polar(theta, r)
plt.title("Galilean Spiral")
plt.show()


# c) Using the same method, make a polar plot of "Fey's function"
# 
# r=e^cos (theta) - 2*cos4 theta +sin ^5 (theta /12)
# 
# in the range 0 </= theta </= 24 pi.

# In[13]:


import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 24*np.pi, 1000)
r = np.exp(np.cos(theta)) - 2 * np.cos(4 * theta) + np.sin(theta/12)**5

ax = plt.subplot(polar=True)
ax.plot(theta, r)

plt.show()


# In[28]:


from pylab import scatter, xlabel, ylabel, xlim, ylim, show
from numpy import loadtxt

data = loadtxt("/Users/bayodeibironke/Downloads/cpresources/stars.txt", float)
x = data[:, 0]
y = data[:, 1]

scatter(x, y)
xlabel("Temperature")
ylabel("Magnitude")
xlim(13000, 0)
ylim(20, -5)
show()


# In[29]:


from pylab import imshow,show 
from numpy import loadtxt

data= loadtxt("/Users/bayodeibironke/Downloads/cpresources/circular.txt",float) 
imshow (data)
show()


# In[4]:


from pylab import imshow,gray,show, bone
from numpy import loadtxt
data= loadtxt("/Users/bayodeibironke/Downloads/cpresources/circular.txt",float) 
imshow(data,origin= "lower")
bone()
show()


# In[5]:


from pylab import imshow,gray,show, bone, hot
from numpy import loadtxt
data= loadtxt("/Users/bayodeibironke/Downloads/cpresources/circular.txt",float) 
imshow(data,origin= "lower")
hot()
show()


# In[7]:


from pylab import imshow,gray,show, bone, hot
from numpy import loadtxt
data= loadtxt("/Users/bayodeibironke/Downloads/cpresources/circular.txt",float) 
imshow(data,origin= "lower",extent=[0,10,0,5] ,aspect=2.0)
gray()
show()


# In[11]:


from math import sqrt,sin,pi
from numpy import empty
from pylab import imshow, gray, show


wavelength= 5.0
k = 2*pi/wavelength
xiO = 1.0
separation = 20.0     # Separation of centers in cm
side = 100.0          # Side of the square in cm
points = 500          # Number of grid points along each side 
spacing = side/points # Spacing of points in cm

# Calculate the positions of the centers of the circles xi side/2 + separation/2
x1 = side/2 + separation/2
y1 = side/2
x2 = side/2 - separation/2
y2 = side/2


# Make an array to store the heights 
xi = empty([points,points] ,float)

# Calculate the values in the array 
for i in range(points):
    y = spacing*i
    for j in range(points):
        for j in range(points):
            x = spacing*j
            r1 = sqrt((x-x1)**2+(y-y1)**2)
            r2 = sqrt((x-x2)**2+(y-y2)**2)
            xi[i,j] = xi0*sin(k*r1) + xi0*sin(k*r2)
         
# Make the plot
imshow(xi, origin = "lower", extent = [0,side,0,side])
gray()
show()


# In[ ]:





# # Exercise 3.3: 
# 
# ## There is a file in the on-line resources called stm. txt, which contains a grid of values from scanning tunneling microscope measurements of the (111} surface of silicon. A scanning tunneling microscope (STM) is a device that measures the shape of a surface at the atomic level by tracking a sharp tip over the surface and measuring quantum tunneling current as a function of position. The end result is a grid of values that represent the height of the surface and the file stm.txt contains just such a grid of values. Write a program that reads the data contained in the file and makes a density plot of the values. Use the various options and variants you have learned about to make a picture that shows the structure of the silicon surface clearly.

# In[18]:


import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file into a 2D array
data = np.loadtxt("/Users/bayodeibironke/Downloads/cpresources/stm.txt")

# Plot the density of the values using a colormap
plt.imshow(data, cmap='hot')

# Add a colorbar to show the scale of the values
plt.colorbar()

# Label the axes
plt.xlabel('X Position')
plt.ylabel('Y Position')

# Show the plot
plt.show()


# In[19]:


import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file into a 2D array
data = np.loadtxt("/Users/bayodeibironke/Downloads/cpresources/stm.txt")

# Plot the density of the values using a colormap
plt.imshow(data, cmap='gray')

# Add a colorbar to show the scale of the values
plt.colorbar()

# Label the axes
plt.xlabel('X Position')
plt.ylabel('Y Position')

# Show the plot
plt.show()


# In[20]:


import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file into a 2D array
data = np.loadtxt("/Users/bayodeibironke/Downloads/cpresources/stm.txt")

# Plot the density of the values using a colormap
plt.imshow(data, cmap='jet')

# Add a colorbar to show the scale of the values
plt.colorbar()

# Label the axes
plt.xlabel('X Position')
plt.ylabel('Y Position')

# Show the plot
plt.show()


# In[23]:


import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file into a 2D array
data = np.loadtxt("/Users/bayodeibironke/Downloads/cpresources/stm.txt")

# Plot the density of the values using a colormap
plt.imshow(data, cmap='hsv')

# Add a colorbar to show the scale of the values
plt.colorbar()

# Label the axes
plt.xlabel('X Position')
plt.ylabel('Y Position')

# Show the plot
plt.show()


# In[24]:


import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file into a 2D array
data = np.loadtxt("/Users/bayodeibironke/Downloads/cpresources/stm.txt")

# Plot the density of the values using a colormap
plt.imshow(data, cmap='bone')

# Add a colorbar to show the scale of the values
plt.colorbar()

# Label the axes
plt.xlabel('X Position')
plt.ylabel('Y Position')

# Show the plot
plt.show()


# In[26]:


import matplotlib.pyplot as plt

# Read data from the stm.txt file
data = [[float(val) for val in line.strip().split()] for line in open("/Users/bayodeibironke/Downloads/cpresources/stm.txt")]

# Plot a density plot of the data with the origin at the lower left corner
plt.imshow(data, origin='lower', cmap='gray')

# Add axis values to the plot using the extent option
plt.xticks(range(len(data[0])), [str(x) for x in range(len(data[0]))])
plt.yticks(range(len(data)), [str(y) for y in range(len(data))])
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("STM Data")

# Change the aspect ratio of the plot to 2.0
plt.gca().set_aspect(2.0)

# Add a color scale next to the figure
plt.colorbar()

# Show the plot
plt.show()

