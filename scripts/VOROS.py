import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier

#code for creating convex hull, setting up limits for integration and evaluating the definite integral

def cull_pnts(pnt_arr):
    """Given an ndarray of points in the unit square, return the convex hull of the ROC curve oriented from (0,0) to (1,1)"""
    pnt_arr=np.concatenate([np.concatenate([np.array([[0,0]]),pnt_arr],axis=0),[[1,1],[1,0]]], axis=0)
    hull=pnt_arr[ConvexHull(pnt_arr).vertices,:]
    upper=hull[np.any(hull!=np.array([1,0]),axis=1)]
    return upper[np.lexsort(upper[:,::-1].T),:].tolist()

def pnts_to_slopes(pnt_lst):
    """Given a collection of ROC points, calculate the slopes of the lines between them"""
    slopes=[]
    pnt_lst=cull_pnts(pnt_lst)
    base=max([pnt[1] for pnt in pnt_lst if pnt[0]==0])
    pnts=[[0,base]]
    for i in range(len(pnt_lst)-1):
        p1=pnt_lst[i]
        p2=pnt_lst[i+1]
        if(p1[0]==p2[0]): #consecutive points on the y-axis don't count.
            continue
        if(p1[1]==p2[1]): #ignore points on the line y=1.
            break
        slope=(p1[1]-p2[1])/(p1[0]-p2[0])
        slopes.append(slope)
        pnts.append(pnt_lst[i+1])
    return pnts,slopes


def pnts_to_lims(pnt_lst):
    """Given a collection of ROC points, calculate the values of t where each point dominates"""
    pnt_lst,slopes = pnts_to_slopes(pnt_lst)
    limits=[]
    limits.append(1) #for classifiers on the y-axis
    for slope in slopes:
        limits.append(slope/(1+slope))
    limits.append(0) #for classifiers on the line y=1
    return [ [pnt_lst[i],limits[i+1],limits[i]] for i in range(len(pnt_lst))]


#debugging function
def area(h,k, t):
    """return the area of lesser classifiers for the point (h,k) at the threshold t, unless a baseline classifier does better, in which case, return the baseline's area."""
    if(t==0):
        if(k!=1):
            print(f"({h},{k}) does not outperform the baseline classifier at t=0, returning baseline area instead")
        return 1
    if(t==1):
        if(h==0):
            print(f"({h},{k}) does not outperform the baseline classifier at t=1, returning baseline area instead")
        return 1
    if(k/h<t/(1-t) or (1-k)/(1-h)<t/(1-t)):
        print(f"({h},{k}) does not dominate at t={t}, returning baseline area instead")
        return 1-max(t,1-t)**2/(t*(1-t))
    return 1+0.5*((1-k)**2+h**2-2*h*(1-k)-(1-k)**2/t-h**2/(1-t))

def antiderivative(h,k,t):
    """Calculate the antiderivative at a point"""
    if(t==0):
        return 0 #log(0) is undefined but the limit exists for ROC points on the line y=1
    if(t==1):
        return 1+( (1-k)**2 )/2 #log(0) is undefined, but the limit exists for ROC points on the y-axis
    else:
        return t+(t/2)*(1-k-h)**2-0.5*( ((1-k)**2) * np.log(t) - (h**2) * np.log(1-t) )
    
def integral(data,interval):
    """Take the average value of the Area of Lesser Classifiers for a piecewise curve (with limits described in data) over the 
    specified interval"""
    total=0
    for i in range(len(data)):
        #if the point's range doesn't intersect the interval, skip it
        if(data[i][1]>interval[1] or data[i][2]<interval[0]):
            continue
        #if the point's range contains the interval, integrate with the integral
        if(interval[1]<=data[i][2] and interval[0]>=data[i][1]):
            a=interval[0]
            b=interval[1]
            #print(a,b,1)
            return (antiderivative(data[i][0][0],data[i][0][1],b)-antiderivative(data[i][0][0],data[i][0][1],a))/(interval[1]-interval[0])
        #If the point's range overlaps with the interval, set limits using the point and the potentially the interval
        if(data[i][1]<interval[0] and data[i][2]>=interval[0]):
            #print("if")
            a=interval[0]
            b=data[i][2]
        elif(data[i][1]<=interval[1] and data[i][2]>interval[1]):
            #print("elif")
            a=data[i][1]
            b=interval[1]
        else:
            #print("else")
            b=data[i][2]
            a=data[i][1]
        #print(a,b,2,total,interval[1],interval[0])
        total+= antiderivative(data[i][0][0],data[i][0][1],b)-antiderivative(data[i][0][0],data[i][0][1],a)
    return total/(interval[1]-interval[0])

def Volume(pnt_lst,interval):
    """Calculate the volume of the ROC curve as a list of points in the given cost interval"""
    return integral(pnts_to_lims(pnt_lst),interval)

def cost_equivalent(pnts1,pnts2):
    """Function for determining for which values of aggregate cost two ROC curves (given as a list of points)
    are cost equivalent, returns a nested list consisting lists of the form [p1,p2,t] where p1 is a point
    on the first curve, p2 is a point on the second curve, and t is the aggregate cost where the
    p1 and p2 offer the same performance."""
    curve1,slopes1=pnts_to_slopes(pnts1)
    curve2,slopes2=pnts_to_slopes(pnts2)
    ans=[]
    for i in range(len(curve1)):
        if(curve1[i][0]==0):
            upper1=np.inf
            lower1=slopes1[i]
        elif(curve1[i][1]==1):
            upper1=slopes1[i-1]
            lower1=0
        else:
            upper1=slopes1[i-1]
            lower1=slopes1[i]
        for j in range(len(curve2)):
            try:
                slope=(curve1[i][1]-curve2[j][1])/(curve1[i][0]-curve2[j][0])
            except ZeroDivisionError:
                slope=-1
            if(slope<0):
                continue
            if(curve2[j][0]==0):
                upper2=np.inf
                lower2=slopes2[j]
            elif(curve2[j][1]==1):
                upper2=slopes2[j-1]
                lower2=0
            else:
                upper2=slopes2[j-1]
                lower2=slopes2[j]
            if(slope<upper1 and slope>lower1 and slope<upper2 and slope>lower2):
                ans.append([curve1[i],curve2[j],slope])
    return ans