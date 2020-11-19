import numpy as np

def wrapToPi(a):
    if isinstance(a, list):
        return [(x + np.pi) % (2*np.pi) - np.pi for x in a]
    return (a + np.pi) % (2*np.pi) - np.pi


def add_padding(padding, data, height, width):
    h=height
    w=width
    img = np.zeros((h, w))
    percentage=100

    # padding=2
	
    for k in range(padding):
        for i in range(0,h):
            for j in range(0,w):
                if data[i*w+j]==100:#obstacle
                    img[i,j]=100 #obstacle
                    img[i+1,j]=max(percentage,img[i+1,j])
                    img[i-1,j]=max(percentage,img[i-1,j])
                    img[i+1,j+1]=max(percentage,img[i+1,j+1])
                    img[i+1,j-1]=max(percentage,img[i+1,j-1])
                    img[i-1,j+1]=max(percentage,img[i-1,j+1])
                    img[i-1,j-1]=max(percentage,img[i-1,j-1])
                    img[i,j+1]=max(percentage,img[i,j+1])
                    img[i,j-1]=max(percentage,img[i,j-1])
                elif data[i*w+j]!=-1:
                    img[i,j]=data[i*w+j] #obstacle with less probability
        data=img.flatten()
    
    return list(img.flatten())

