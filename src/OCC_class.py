
# from functions_plot import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import expit
# from bico.core import BICO
# from bico.geometry.point import Point
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

class OCC:

    def __init__(self, X, step = 0.05, eps = 0.005, labels=None, y=None):
        import matplotlib.pyplot as plt
        import numpy as np

        self.X = X
        self.n = X.shape[0]
        self.y=y
        self.figsize = (10,5)

        # plot parameters
        self.step = step
        self.eps = eps
        self.norm_colors = mpl.colors.Normalize(vmin=0,vmax=100)
        # self.set_grid()
        if labels != None:
            self.labels=labels

        pass

    # Plot functions

    def plot(self):
        plt.scatter(self.X[:,0], self.X[:,1], edgecolors  = 'black')
        plt.xlim((min(self.X[:,0])-self.step,max(self.X[:,0])+self.step))
        plt.ylim((min(self.X[:,1])-self.step,max(self.X[:,1])+self.step))
        pass

    def plotNew(self,row, col):
        plt.scatter(self.X[:,row], self.X[:,col], edgecolors  = 'purple')
        # plt.scatter(self.X[:,row], self.X[:,col], edgecolors  = 'black')
        plt.xlim((min(self.X[:,row])-self.step,max(self.X[:,row])+self.step))
        plt.ylim((min(self.X[:,col])-self.step,max(self.X[:,col])+self.step))
        plt.xlabel(list(self.labels.values())[row])
        plt.ylabel(list(self.labels.values())[col])
        pass

    def set_grid(self):
        self.x_axis = np.arange(min(self.X[:,0])-self.step,max(self.X[:,0])+2*self.step,self.step)
        if self.X.shape[1] > 1:
            self.y_axis = np.arange(min(self.X[:,1])-self.step,max(self.X[:,1])+2*self.step,self.step)
        else:
            self.y_axis = np.arange(min(self.X[:,0])-self.step,max(self.X[:,0])+2*self.step,self.step)

        self.my_grid = []
        for i in self.x_axis:
            for j in self.y_axis:
                self.my_grid.append([i,j])
        self.my_grid = np.array(self.my_grid)

    def setGrid(self, row, col):
        self.x_axis = np.arange(min(self.X[:,row])-self.step,max(self.X[:,row])+2*self.step,self.step)
        self.y_axis = np.arange(min(self.X[:,col])-self.step,max(self.X[:,col])+2*self.step,self.step)

        self.my_grid = []
        for i in self.x_axis:
            for j in self.y_axis:
                self.my_grid.append([i,j])
        self.my_grid = np.array(self.my_grid)
        pass


    def Normalize_Pred(self):
        self.pred_100 = self.pred -np.min(self.pred)
        self.pred_100 = (self.pred_100 / np.max(self.pred_100)) * 100
        pass

    def set_delim(self):
        ix_delim = np.where((self.pred  < self.eps) & (self.pred  > -self.eps))[0]
        self.x_delim = []
        self.y_delim = []
        for i in range(len(ix_delim)):
            a, b = np.divmod(ix_delim[i],len(self.y_axis))
            self.x_delim.append(self.x_axis[a])
            self.y_delim.append(self.y_axis[b])
        pass

    def plot_pred(self, pred):

        X_axis, Y_axis = np.meshgrid(self.x_axis, self.y_axis)
        C = np.transpose(pred.reshape((len(self.x_axis), len(self.y_axis))))
        plt.pcolor(X_axis, Y_axis, C, norm = self.norm_colors, cmap = 'YlGnBu')
        # plt.pcolor(X_axis, Y_axis, C, norm = self.norm_colors, cmap = 'YlOrRd')
        plt.scatter(self.x_delim, self.y_delim, c = 'black', s = 10)
        self.plot()
        pass

    def plotPred(self, pred, row, col):

        X_axis, Y_axis = np.meshgrid(self.x_axis, self.y_axis)
        C = np.transpose(pred.reshape((len(self.x_axis), len(self.y_axis))))
        plt.pcolor(X_axis, Y_axis, C, norm = self.norm_colors, cmap = 'YlOrRd')
        plt.scatter(self.x_delim, self.y_delim, c = 'black', s = 10)
        self.plotNew(row,col)
        pass

    def plot_pred_plan(self):
        self.pred = self.predict(self.my_grid)
        self.pred_bin = np.sign(self.pred)
        self.set_delim()
        self.Normalize_Pred()

        plt.figure(figsize=self.figsize)
        plt.subplot(1,2,1)
        self.plot_pred(self.pred_100)
        plt.subplot(1,2,2)
        self.plot_pred((self.pred_bin+1)*100)
        pass

    def plotPredPlan(self, row, col):
        self.setGrid(row, col)
        self.pred = self.predictNew(self.my_grid, row, col)
        self.pred_bin = np.sign(self.pred)
        self.set_delim()
        self.Normalize_Pred()

        plt.figure(figsize=self.figsize)
        plt.subplot(1,2,1)
        self.plotPred(self.pred_100, row, col)
        plt.subplot(1,2,2)
        self.plotPred((self.pred_bin+1)*100, row,col)
        plt.show() #ASM
        pass

    # Skeletons for fit and predict
    def fit(self):
        pass

    def predict(self):
        pass

    # Others methods
    def RBF_Kernel(self, X, sigma2, Y = None):
        " Compute the RBF kernel matrix of X"
        from sklearn.metrics.pairwise import euclidean_distances

        if type(Y)==type(None):
            Y = X

        K = euclidean_distances(X,Y, squared=True)
        K *= -1./sigma2
        K = np.exp(K)
        return K


    def RBF_KernelNew(self, X, sigma2, row, col, Y = None):
        " Compute the RBF kernel matrix of X"
        from sklearn.metrics.pairwise import euclidean_distances

        if type(Y)==type(None):
            Y = X

        K = euclidean_distances(X[:,[row,col]],Y, squared=True)
        K *= -1./sigma2
        K = np.exp(K)
        return K

from sklearn.cluster import DBSCAN
class dbscan(OCC):

    def fit_mod(self):
        # self.dbscan =DBSCAN( eps = .2, metric="cosine", min_samples = 5, n_jobs = -1).fit(self.X)
        self.dbscan =DBSCAN( eps = .2, metric="euclidean", min_samples = 5, n_jobs = -1).fit(self.X)

    def fit(self, X=None, y=None):
        self.X=X
        # self.dbscan =DBSCAN( eps = .2, metric="cosine", min_samples = 5, n_jobs = -1).fit(self.X)
        self.dbscan =DBSCAN( eps = .2, metric="euclidean", min_samples = 5, n_jobs = -1).fit(self.X)

    def predict(self, newData):
        pred = self.dbscan.fit_predict(newData) #for anomaly score
        return pred

    def predict_anomaly(self, newData):
        pred=self.predict(newData)
        pred_anom=np.zeros(len(pred))
        anom_index = np.where(pred == -1)
        pred_anom[anom_index]=1
        return pred_anom.astype(int)

    def decision_function(self, newdata):
        return self.predict(newdata)  # for anomaly score

from sklearn.ensemble import IsolationForest

class cisof(OCC):

    def fit_mod(self):
        self.isof =IsolationForest(max_samples=100,random_state=np.random.RandomState(0), contamination=.1) .fit(self.X)

    def fit(self,X=None, y=None):
        self.X=X
        self.isof =IsolationForest(max_samples=100,random_state=np.random.RandomState(0), contamination=.1) .fit(self.X)

    def predict(self, newData):
        pred = self.isof.predict(newData) #for anomaly score
        return pred

    def predict_anomaly(self, newData):
        return (1-(self.predict(newData)+1)/2).astype(int)

    def decision_function(self, newdata):
        return self.predict(newdata)  # for anomaly score

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class optKmeans(OCC):
    def fit(self, max_clusters=5, option="sos", y=None): #"silhoutte"):
        km_metric = []
        K = range(2, max_clusters)  ####  15 was  taken  arbitaorily  to see  where  our elbow touches
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(self.X)
            if option=="sos":
                km_metric.append(km.inertia_)
            if option == "silhoutte":
                label = km.labels_
                sil_coeff = silhouette_score(self.X, label, metric='euclidean')
                km_metric.append(sil_coeff)
        if option == "sos":
            opt_n_clstr=np.argmin(km_metric)+1
        if option == "silhoutte":
            opt_n_clstr=np.argmax(km_metric)+1
        # print("optimum number of clusters "+ str(opt_n_clstr))
        self.optkm=KMeans(n_clusters=opt_n_clstr).fit(self.X)
    def predict(self, data):
        '''
        Anomaly Detection
        1. The small clusters with less than a threshold: A dictionary with clusters as keys and count of number of points as values is created. A threshold of 1% of the dataset  is
           calculated and based on the value count in the dictionary, anomalies are detected.
        2. Isolation data points not belong to any cluster: Based on above created dictionary, the clusters with values as 1 as detected as isolation points.
        3. A data point belongs to a cluster with more than 2 standard deviations: Mean, SD and Mean ± 2*SD for each dimension is calculated at cluster level and points not in this
           range as detected as anomalies.
        '''
        # create an anomaly array
        anomaly_array = []
        threshold = int(0.1 * len(data))
        # # create array of final dataframe
        # final_arr = final_df.values
        # dictionary with cluster and count of number of points
        cluster_len = {}
        final_dict={}
        for index, clustr in enumerate(self.optkm.labels_):
            if clustr in final_dict:
                final_dict[clustr].append(index)
            else:
                final_dict[clustr]=[index]
        for i, c in enumerate(self.optkm.labels_):
            if c in cluster_len:
                cluster_len[c] += 1
                # if c in final_dict:
                #     final_dict[c].append(i)
                # else:
                #     final_dict[c]=i
            else:
                cluster_len[c] = 1
                # if c in final_dict:
                #     final_dict[c].append(i)
                # else:
                #     final_dict[c]=i
        for clster, len_val in cluster_len.items():
            # detect isolation points
            if len_val == 1:
                anomaly_array.append(final_dict[clster])
            # detect clusters with points less than threshold
            elif len_val < threshold:
                for values in final_dict[clster]:
                    anomaly_array.append(values)
        X_dist = self.optkm.transform(self.X) ** 2
        sq_distance=X_dist.sum(axis=1).round(2)
        anamoly_dict = {}
        for clu_key in final_dict.keys():
            anamoly_dict[clu_key] = np.mean(sq_distance[final_dict[clu_key]]) - 2 * np.std(sq_distance[final_dict[clu_key]])
            # anamoly_dict[clu_key] = [np.mean(sq_distance[final_dict[clu_key]]) - 2 * np.std(sq_distance[final_dict[clu_key]]),
            #     np.mean(sq_distance[final_dict[clu_key]]) + 2 * np.std(sq_distance[final_dict[clu_key]])]
        z = 0
        for key1 in set(anamoly_dict.keys()) & set(final_dict.keys()):
            for value1 in final_dict[key1]:
                # check if each point is within the mean ± 2*sd range
                # if ~(np.less(np.array(sq_distance[value1]), np.array(anamoly_dict[key1][z])).any()) and np.greater(
                #         np.array(np.array(sq_distance[value1])), anamoly_dict[key1][z + 1]).all():
                if sq_distance[value1] > anamoly_dict[key1]:
                    anomaly_array.append(value1)
                else:
                    continue
        #
        # anomalies_array = [[np.round(float(i), 2) for i in nested] for nested in anomaly_array]
        #
        # if len(anomalies_array) == 0:
        #     print("no anomalies")
        # else:
        #     print(anomalies_array)
        #     # for anomaly in anomalies_array:
        #     #     print(anomaly)
        # if len(anomaly_array) == 0:
        #     print("no anomalies")
        # else:
        #     print(anomaly_array)
        pred=np.zeros(len(data))
        for i in anomaly_array:
            pred[int(i)] = 1
        return pred
    def predict_anomaly(self, newdata):
        return self.predict(newdata).astype(int)

class OCSVM(OCC):
    def fit_mod(self):
        self.osvmn = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma = 0.001).fit(self.X)
    def fit(self, X=None, y=None):
        self.X=X
        return svm.OneClassSVM(nu=0.2, kernel="rbf", gamma = 0.001).fit(self.X)
    def predict(self, newdata):
        pred = self.osvmn.predict(newdata) #for anomaly score
        return pred
    def predict_anomaly(self, newdata):
        return (1 - (self.predict(newdata) + 1) / 2).astype(int)
    # return np.sign(self.predict(newdata)).astype(int)
    def decision_function(self, newdata):
        return self.predict(newdata)  # for anomaly score


class OSVM(OCC):
    " Support Vector Method for Novelty Detection       Bernhard Schokopf, Robert Williamsonx, Alex Smolax, John Shawe-Taylory, John Platt       Quadratic Programming "

    def fit(self, nu, sigma2,weights):
        from cvxopt import solvers, lapack, matrix, spmatrix
        solvers.options['show_progress'] = 0

        self.X = np.repeat(self.X, repeats=weights, axis=0)
        self.n = self.X.shape[0]

        n = self.n
        self.sigma2 = sigma2
        K = self.X

        P = matrix(K, tc = 'd')
        q = matrix([0]*n, tc = 'd')
        G = matrix(np.concatenate([np.eye(n),-np.eye(n)], axis = 0), tc = 'd')
        h_value = [1./(n * nu)]*n
        h_value.extend([0]*n)
        h = matrix(h_value, tc = 'd')
        A = matrix(np.reshape([1]*n, (1,n)), tc = 'd')
        b = matrix(1, tc = 'd')

        sol = solvers.qp(P,q,G,h,A,b)
        self.alpha = np.array(sol['x'])
        ix_in = np.where((self.alpha > 1e-5) & (self.alpha < 1./(n*nu)))[0][0]
        self.rho = np.inner(np.reshape(self.alpha, (1,n)), K[:,ix_in])
        pass

    def predict(self, newData):
        K = self.RBF_Kernel(self.X, self.sigma2, newData)
        #return np.transpose(np.dot(np.reshape(self.alpha, (1,self.n)), K) - self.rho) #for anomaly
        return np.transpose(expit(np.dot(np.reshape(self.alpha, (1,self.n)), K) - self.rho)) #for anomaly score

    def predictNew(self, newData, row, col):
        K = self.RBF_KernelNew(self.X, self.sigma2, row=row, col=col, Y=newData)
        return np.transpose(np.dot(np.reshape(self.alpha, (1,self.n)), K) - self.rho)

    def predict_anomaly(self, newData):
        return np.sign(np.transpose(self.predict(newData))[0]).astype(int)



class OSVM2(OCC):
    " Simple Incremental One-Class Support Vector Classification       Kai Labusch, Fabian Timm, and Thomas Martinetz "

    def RBF_Kernel(self, X, sigma2, Y = None):
        " Compute the RBF kernel matrix of X"
        from sklearn.metrics.pairwise import euclidean_distances

        if type(Y)==type(None):
            Y = X

        K = euclidean_distances(X,Y, squared=True)
        K *= -1./sigma2
        K = np.exp(K)
        return K

    def RBF_KernelNew(self, X, sigma2, row, col, Y = None):
        " Compute the RBF kernel matrix of X"
        from sklearn.metrics.pairwise import euclidean_distances

        if type(Y)==type(None):
            Y = X

        K = euclidean_distances(X[:,[row,col]],Y, squared=True)
        K *= -1./sigma2
        K = np.exp(K)
        return K

    def fit(self, nu, sigma2, nbIter = 1000):

        n = self.n
        self.sigma2 = sigma2

        K = self.RBF_Kernel(self.X, sigma2)
        alpha = np.zeros(n)
        count = 0
        Z = self.X.copy()
        i_min = np.random.choice(range(n), 1)
        i_max = np.random.choice(range(n), 1)
        while (i_max == i_min):
            i_max = np.random.choice(range(n), 1)

        while count < nbIter:
            count = count + 1
            K = self.RBF_Kernel(Z, 1)
            h = np.dot(K + (n*nu)/2 * np.eye(n), alpha)
            i_min2 = np.argmin(h)
            ix_in = np.where(alpha > 1e-7)[0]
            if len(ix_in) > 0:
                i_max2 = ix_in[np.argmax(h[ix_in])]
            else:
                i_max2 = np.argmax(h)
            Z[i_min,:] = Z[i_min2,:]
            Z[i_max,:] = Z[i_max2,:]
            i_min = i_min2
            i_max = i_max2
            i_min_alpha = np.argmin(alpha)
            i_max_alpha = np.argmax(alpha)
            alpha[i_min_alpha] = alpha[i_min_alpha]+2
            alpha[i_max_alpha] = alpha[i_max_alpha]-1

        h = np.dot(K + (n*nu)/2 * np.eye(n), alpha)
        self.alpha = alpha / np.sqrt(np.dot(alpha, h))
        h = np.dot(K + (n*nu)/2 * np.eye(n), self.alpha)
        self.rho = min(h)
        pass

    def predict(self, newData):
        K = self.RBF_Kernel(self.X, self.sigma2, newData)
        #return np.transpose(np.dot(np.reshape(self.alpha, (1,self.n)), K) - self.rho) #for anomaly
        return np.transpose(expit(np.dot(np.reshape(self.alpha, (1,self.n)), K) - self.rho)) #for anomaly score


    def predictNew(self, newData, row, col):
        K = self.RBF_KernelNew(self.X, self.sigma2, row=row, col=col, Y=newData)
        return np.transpose(np.dot(np.reshape(self.alpha, (1,self.n)), K) - self.rho)

    def predict_anomaly(self, newData):
        return np.sign(np.transpose(self.predict(newData))[0]).astype(int)


class OkNN(OCC):

    def fit(self, thresh, k, kernel = False):
        self.thresh = thresh
        self.k = k
        self.kernel = kernel

        pass

    def predict(self, newData):
        from sklearn.metrics.pairwise import euclidean_distances

        n = self.n
        n2 = newData.shape[0]

        if self.kernel:
            K = self.RBF_Kernel(self.X, 1)
            K2 = np.transpose(self.RBF_Kernel(self.X, 1, newData))
            D = euclidean_distances(K, K2, squared=True) # (n, len(newData))
        else:
            D = euclidean_distances(self.X, newData, squared=True) # (n, len(newData))
        # Find k nearest neighbours for each column
        col_range = range(D.shape[1])

        if self.k==1:
            ix_nns = np.argsort(D, axis = 0)[:2, :]
        else:
            ix_nns = np.argsort(D, axis = 0)[:self.k, :]
        ix_nn = ix_nns[0,:]
        D1 = D[ix_nn, col_range]
        D2 = np.mean(D[ix_nns, col_range], axis = 0)
        # #this is for prediction of anomaly
        # pred = np.array(D1 < D2*self.thresh, dtype = int) #corrections for D2=0 case
        #the following gives anomaly score
        pred = np.array(expit(D1-self.thresh))
        # pred = - D1/D2 + self.thresh
        return pred


    def predictNew(self, newData, row, col):
        from sklearn.metrics.pairwise import euclidean_distances

        n = self.n
        n2 = newData.shape[0]

        if self.kernel:
            K = self.RBF_Kernel(self.X, 1)
            K2 = np.transpose(self.RBF_KernelNew(self.X, 1, row=row, col=col, Y=newData))
            D = euclidean_distances(K, K2, squared=True) # (n, len(newData))
        else:
            D = euclidean_distances(self.X, newData, squared=True) # (n, len(newData))
        # Find k nearest neighbours for each column
        col_range = range(D.shape[1])

        if self.k==1:
            ix_nns = np.argsort(D, axis = 0)[:2, :]
        else:
            ix_nns = np.argsort(D, axis = 0)[:self.k, :]
        ix_nn = ix_nns[0,:]
        D1 = D[ix_nn, col_range]
        D2 = np.mean(D[ix_nns, col_range], axis = 0)
        #pred = np.array(D1/D2 < self.thresh, dtype = int)
        #pred = - D1/D2 + self.thresh #for anomaly
        pred = expit(D1 - self.thresh) #for anomaly score
        return pred

    def predict_anomaly(self, newData):
        return np.sign(self.predict(newData)).astype(int)

class dkNN(OCC):

    def fit(self, thresh, k, kernel = False):
        self.thresh = thresh
        self.k = k
        self.kernel = kernel

        pass

    def predict(self, newData, n_bucket=10):
        from sklearn.metrics.pairwise import cosine_similarity
        import heapq

        idxs = []
        dists = []
        buckets = np.array_split(self.X, n_bucket)

        if self.kernel:
            K = self.RBF_Kernel(self.X, 1)
            K2 = np.transpose(self.RBF_Kernel(self.X, 1, newData))
            for b in range(n_bucket):
                K = self.RBF_Kernel(buckets[b], 1)
                K2 = np.transpose(self.RBF_Kernel(buckets[b], 1, newData))
                cosim = cosine_similarity(K, K2)
                idx0 = [(heapq.nlargest((self.k + 1), range(len(i)), i.take)) for i in cosim]
                idxs.extend(idx0)
                dists.extend([cosim[i][idx0[i]] for i in range(len(cosim))])
        else:
            for b in range(n_bucket):
                cosim = cosine_similarity(buckets[b], newData)
                idx0 = [(heapq.nlargest((self.k + 1), range(len(i)), i.take)) for i in cosim]
                idxs.extend(idx0)
                dists.extend([cosim[i][idx0[i]] for i in range(len(cosim))])
        # Find k nearest neighbours for each column
        col_range = range(dists.shape[1])

        if self.k==1:
            ix_nns = np.argsort(dists, axis = 0)[:2, :]
        else:
            ix_nns = np.argsort(dists, axis = 0)[:self.k, :]
        ix_nn = ix_nns[0,:]
        D1 = dists[ix_nn, col_range]
        D2 = np.mean(dists[ix_nns, col_range], axis = 0)
        # #this is for prediction of anomaly
        # pred = np.array(D1 < D2*self.thresh, dtype = int) #corrections for D2=0 case
        #the following gives anomaly score
        pred = np.array(expit(D1-self.thresh))
        # pred = - D1/D2 + self.thresh
        return pred

    def predict_anomaly(self, newData,n_bucket=10):
        return np.sign(self.predict(newData,n_bucket)).astype(int)

class OkMeans(OCC):
    " Visual Object Recognition through One-Class Learning       QingHua Wang, Luís Seabra Lopes, and David M. J. Tax "

    def fit(self, thresh, k, sample_weight , kernel = False):
        from sklearn.cluster import KMeans

        self.thresh = thresh
        self.k = k
        self.kernel = kernel
        if kernel:
            K = self.RBF_Kernel(self.X, 1)
        else:
            K = self.X

        kmeans = KMeans(n_clusters=k).fit(K,sample_weight=sample_weight)
        self.centers = kmeans.cluster_centers_

        pass
        if kernel:
            K = self.RBF_Kernel(self.X, 1)

    def predict(self, newData):
        from sklearn.metrics.pairwise import euclidean_distances

        if self.kernel:
            newData = np.transpose(self.RBF_Kernel(self.X, 1, newData))
        D = euclidean_distances(newData, self.centers, squared=True)
        D = np.min(D, axis = 1)
        #pred = -D + self.thresh #for anomaly
        pred = expit(D - self.thresh) #for anomaly score
        return pred

    def predictNew(self, newData, row, col):
        from sklearn.metrics.pairwise import euclidean_distances
        if self.kernel:
            newData = np.transpose(self.RBF_Kernel(self.X, 1, newData))
        D = euclidean_distances(newData, self.centers[:,[row,col]], squared=True)
        D = np.min(D, axis = 1)
        pred = -D + self.thresh
        return pred


    def predict_anomaly(self, newData):
        return np.sign(self.predict(newData)).astype(int)

class dkMeans(OCC):

    def fit(self, thresh, k, kernel = False):
        from sklearn.cluster import KMeans

        self.thresh = thresh
        self.k = k
        self.kernel = kernel
        if kernel:
            K = self.RBF_Kernel(self.X, 1)
        else:
            K = self.X

        kmeans = KMeans(n_clusters=k).fit(K)
        self.centers = kmeans.cluster_centers_

        pass
        if kernel:
            K = self.RBF_Kernel(self.X, 1)

    def predict(self, newData, n_bucket=1):
        from sklearn.metrics.pairwise import cosine_similarity
        import heapq

        idxs = []
        dists = []
        buckets = np.array_split(self.X, n_bucket)
        for b in range(n_bucket):
            if self.kernel:
                newData = np.transpose(self.RBF_Kernel(buckets[b], 1, newData))
            cosim = cosine_similarity(newData, self.centers)
            idx0 = [(heapq.nlargest((self.k + 1), range(len(i)), i.take)) for i in cosim]
            idxs.extend(idx0)
            dists.extend([cosim[i][idx0[i]] for i in range(len(cosim))])
        D = np.min(dists, axis = 1)
        #pred = -D + self.thresh #for anomaly
        pred = expit(D - self.thresh) #for anomaly score
        return pred


    def predict_anomaly(self, newData,n_bucket=10):
        return np.sign(self.predict(newData,n_bucket)).astype(int)

#supervised models
#supervised models
#For supervised models


class OLR(OCC):
    from sklearn.linear_model import LogisticRegression

    def fit_mod(self):
        self.lr=LogisticRegression(random_state=0, class_weight='balanced').fit(self.X, self.y)

    def fit(self, X=None, y=None):
        self.X=X
        self.y=y
        self.lr = LogisticRegression(random_state=0, class_weight='balanced').fit(self.X, self.y)

    def predict(self, newData):
        #pred = self.lr.predict(newData) #for anomaly
        pred = pred = self.lr.predict_proba(newData)[:,1] #for anomaly score
        return pred

    def predictNew(self, newData, row, col):
        pred = self.lr.predict(newData)
        return pred


    def predict_anomaly(self, newData):
        return (self.predict(newData)>0.5).astype(int)

class ORF(OCC):
    def fit_mod(self):
        self.rf=RandomForestClassifier(max_depth=2, random_state=0, class_weight='balanced').fit(self.X, self.y)
    def fit(self, X=None, y=None):
        self.X=X
        self.y=y
        self.rf=RandomForestClassifier(max_depth=2, random_state=0, class_weight='balanced').fit(self.X, self.y)
    def predict(self, newData):
        pred = self.rf.predict_proba(newData)[:,1] #for anomaly score
        return pred
    def predictNew(self, newData, row, col):
        pred = self.rf.predict(newData)
        return pred
    def predict_anomaly(self, newData):
        return (self.predict(newData)>0.5).astype(int)

class OGBM(OCC):
    def fit_mod(self):
        self.gbm=GradientBoostingClassifier(n_estimators=20, learning_rate=0.05, max_features=2, max_depth=2, random_state=0).fit(self.X, self.y)
    def fit(self, X=None, y=None):
        self.X=X
        self.y=y
        self.gbm=GradientBoostingClassifier(n_estimators=20, learning_rate=0.05, max_features=2, max_depth=2, random_state=0).fit(self.X, self.y)
    def predict(self, newData):
        pred = self.gbm.predict_proba(newData)[:,1] #for anomaly score
        return pred
    def predictNew(self, newData, row, col):
        pred = self.gbm.predict(newData)
        return pred
    def predict_anomaly(self, newData):
        return (self.predict(newData)>0.5).astype(int)

class OSVC(OCC):

    def fit_mod(self):
        self.linSVM=SVC(probability=True, class_weight='balanced').fit(self.X, self.y)

    def fit(self, X=None, y=None):
        self.X=X
        self.y=y
        self.linSVM=SVC(probability=True, class_weight='balanced').fit(self.X, self.y)

    def predict(self, newData):
        pred = self.linSVM.predict_proba(newData)[:,1] #for anomaly score
        return pred

    def predictNew(self, newData, row, col):
        pred = self.linSVM.predict(newData)
        return pred

    def predict_anomaly(self, newData):
        return (self.predict(newData)>0.5).astype(int)



# X = 0.3 * np.random.randn(100, 2)
# X = np.r_[X + 2, X - 2]
# plot(X)

# osvm = OSVM(X)
# nu = 0.001
# sigma2 = 20
# osvm.fit(nu, sigma2)
# osvm.plot_pred_plan()

# osvm2 = OSVM2(X)
# nu = 0.001
# sigma2 = 5
# osvm2.fit(nu, sigma2)
# osvm2.plot_pred_plan()

# oknn = OkNN(X)
# oknn.fit(thresh = 0.9, k = 1)
# oknn.plot_pred_plan()

# oknn = OkNN(X)
# oknn.fit(thresh = 0.9, k = 1, kernel = True)
# oknn.plot_pred_plan()

# okmeans = OkMeans(X)
# okmeans.fit(thresh = 10, k = 1)
# okmeans.plot_pred_plan()

# okmeans = OkMeans(X)
# okmeans.fit(thresh = 20, k = 1, kernel = True)
# okmeans.plot_pred_plan()

# ### Variant of SMO (Sequential Minimal Optimization) : NE MARCHE PAS

# def get_C(alpha, K, i, j):
#     ix = [x for x in range(len(alpha)) if x not in [i,j]]
#     amp_alpha = alpha[ix]
#     amp_K = K[ix,:][:,ix]
#
#     return np.dot(amp_alpha, np.dot(amp_K, amp_alpha))
#
# def get_Os(alpha, K, i, j):
#     Os = K[:,i] * alpha[i] + K[:,j] * alpha[j] + get_Cs(alpha, K, i, j)
#     return Os
#
# def get_Cs(alpha, K, i, j):
#     ix = [x for x in range(len(alpha)) if x not in [i,j]]
#     amp_alpha = alpha[ix]
#     amp_K = K[:,ix]
#     Cs = np.dot(amp_K, amp_alpha)
#     return Cs
#
# def is_support(alpha, u_bound, tol = 1e-7):
#     return (alpha >= tol) & (alpha <= -tol + u_bound)
#
# def update_rho(alpha, K, nu, u_bound):
#     n = len(alpha) # n = self.n
#     #nu = self.nu
#     ix_support = np.where(is_support(alpha, u_bound))[0][0]
#     rho = np.inner(np.reshape(alpha, (1,n)), K[:,ix_support])
#     return rho
#
# def update_alpha(alpha, K, i, j, u_bound):
#     ix = [x for x in range(len(alpha)) if x not in [i,j]]
#     amp_alpha = alpha[ix]
#     delta = 1 - np.sum(amp_alpha)
#     #Ci, Cj = get_Cs(alpha, K, i, j)
#     #alpha[j] = (delta * (K[i,i] - K[i,j]) + Ci - Cj) / (K[i,i] + K[j,j] - 2*K[i,j])
#     Os = get_Os(alpha, K, i, j)
#     alpha[j] = alpha[j] + (Os[i] - Os[j]) / (K[i,i] + K[j,j] - 2*K[i,j])
#     alpha[i] = delta - alpha[j]
#     print alpha[i], alpha[j]
#     alpha = project_alpha(alpha, i, j, delta, u_bound)
#     print alpha[i], alpha[j]
#     return alpha
#
# def project_alpha(alpha, i, j, delta, u_bound):
#     if (is_support(alpha[i], u_bound, tol = 0) & is_support(alpha[j], u_bound, tol = 0)):
#         return alpha
#     else:
#         alpha[j] = max(0, min(min(delta,u_bound), alpha[j]))
#         alpha[i] = max(0, min(u_bound, delta - alpha[j]))
#         return alpha

# if False:
#     ### Initialization
#     nu = 0.12
#     n = X.shape[0]
#
#     # Initialization of alpha
#     u_bound = 1./(nu*n)
#     alpha = np.zeros(n)
#     ix_non_zero = np.random.choice(range(n), int(np.floor(nu*n)), False)
#     alpha[ix_non_zero] = u_bound
#     if type(nu*n) != int:
#         ix = np.where(alpha == 0)[0]
#         ix = np.random.choice(ix, 1)
#         alpha[ix] = 1-np.sum(alpha)
#         ix_non_zero = np.append(ix_non_zero, ix)
#
#     # Initialization of rho
#     K = osvm.RBF_Kernel(X, 1)
#     i = np.arange(n)[ix_non_zero[0]]
#     j = np.arange(n)[ix_non_zero[1]]
#     Os = get_Os(alpha, K, i, j)
#     rho = np.max(Os[ix_non_zero])
#     alpha = update_alpha(alpha, K, i, j, u_bound)
#     rho = update_rho(alpha, K, nu, u_bound)
#
#
#     ### Optimization
#     cond1 = (Os - rho) * alpha > 1e-7
#     cond2 = (rho - Os) * (u_bound - alpha) > 1e-7
#     cond = cond1 | cond2
#     count = 0
#     while(sum(cond) > 1):
#         count = count +1
#         i = np.where(cond)[0][0]
#         ix_support = np.where(is_support(alpha, u_bound))[0]
#         j = ix_support[np.argmax(np.abs(Os[i] - Os[ix_support]))]
#         alpha = update_alpha(alpha, K, i, j, u_bound)
#         rho = update_rho(alpha, K, nu, u_bound)
#
#         cond1 = (Os - rho) * alpha > 1e-7
#         cond2 = (rho - Os) * (u_bound - alpha) > 1e-7
#         cond = cond1 | cond2
#
#         if count > 10000:
#             break