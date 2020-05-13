from scipy.io import loadmat
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys

# # QUESTION 1

# In[2]:

question = sys.argv[1]

def alp_kumbasar_21602607_HW3(question):
#mat = loadmat("hw3_data2.mat")

    if question == '1':
        print("Question 1 is running:")
        with h5py.File('hw3_data2.mat', 'r') as f:
            yn = np.array(list(f['Yn']))
            xn = np.array(list(f['Xn']))


        # In[3]:


        print("Xn array shape is: " + str(xn.shape))
        print("Yn array shape is: " + str(yn.shape))


        # In[4]:


        xn = xn.T
        yn = yn.flatten()


    # In[5]:


    #print(yn.shape)


    # ## Question 1 A
        print("Question 1 A")
    # In[6]:


        def ridge_regression(x, y, lamda):
            
            ik = np.eye(np.shape(x)[1])
            w1 = np.dot(np.linalg.inv(np.dot(x.T, x) + lamda * ik), x.T)
            wf = np.dot(w1, y)
            
            return wf


        # In[7]:


        def r2_finder(y,pred):
            
            p = np.corrcoef(y, pred)[0,1]
            r2 = p ** 2

            return r2


        # In[8]:


        def cross_val(x,y,lamda,kFold):
            
            size = np.shape(y)[0]
            train_size = int(size * 8/10)
            val_size = int(size * 1/10)
            test_size = int(size * 1/10)
            res_test = np.zeros((10,lamda.shape[0]))
            res_val = np.zeros((10,lamda.shape[0]))
            
            for i in range(kFold):
                    
                val_ind = np.arange(i*val_size, (i+1)*val_size) %size
                test_ind = np.arange((i+1)*test_size, (i+2)*test_size) %size
                train_ind = np.arange((i+2)*test_size, (i+2)*test_size + train_size) %size

                val_f = x[val_ind]
                val_l = y[val_ind]

                train_f = x[train_ind]
                train_l = y[train_ind]

                test_f = x[test_ind]
                test_l = y[test_ind]
                counter = 0
                for lam in lamda:
                    
                    weight = ridge_regression(train_f, train_l, lam)
                    
                    val_res = r2_finder(val_l, val_f.dot(weight).flatten())
                    test_res = r2_finder(test_l, test_f.dot(weight).flatten())
                    
                    res_test[i, counter] = test_res
                    res_val[i, counter] = val_res
                    counter += 1
                print(i)
                    
            return res_test, res_val


        # In[9]:


        lamda_arr = np.logspace(0, 12, num=600, base=10)


        # In[10]:


        result_test, result_val = cross_val(xn,yn,lamda_arr,10)
        #print(result_test.shape)


        # In[11]:


        summer_val = 0
        summer_test = 0
        means = np.zeros((2,lamda_arr.shape[0]))

        for i in range(lamda_arr.shape[0]):
            summer_test = 0
            summer_val = 0
            for j in range(10):
                summer_test = result_test[j,i] + summer_test
                summer_val = result_val[j,i] + summer_val
            
            means[0,i] = summer_test/10
            means[1,i] = summer_val/10
            
        for i in range(lamda_arr.shape[0]):
            means[0,i] = np.sum(result_test[:,i])
            means[1,i] = np.sum(result_val[:,i])


        # In[12]:


        print(np.argmax(means[0]))
        #np.where(means[0]==0.012337268595244607)


        # In[13]:


        print(means.shape)


        # In[14]:


        
        plt.plot(np.arange(600), means[0])
        plt.plot(np.arange(600), means[1])
        plt.xticks([0, 100, 200, 300, 400, 500, 600], ["0", "10^2", "10^4", "10^6", "10^8", "10^10", "10^12"])
        plt.xlabel("Lambda Value")
        plt.ylabel("Mean of R^2 performance value")
        plt.legend(["TEST RESULT", "VALIDATION RESULT"])
        plt.show()


        # In[15]:


        l_optimal = lamda_arr[np.argmax(means[0])]
        print("Optimal lambda found is: " + str(lamda_arr[np.argmax(means[0])]))
        print("R2 in test is: " + str(means[0,np.argmax(means[0])]))
        print("R2 in validation is: " + str(means[1,np.argmax(means[0])]))


    # # Question 1 B

    # In[16]:
        print("Question 1 B")

        ITER_NO = 500
        size = 1000
        weight_1b_OLS = []
        np.random.seed(7)
        for i in range(ITER_NO):
            
            random = np.random.choice(np.arange(size), size)
            
            y_b = yn[random]
            x_b = xn[random]
            
            w_holder = ridge_regression(x_b, y_b, 0)
            
            weight_1b_OLS.append(w_holder)


        weight_1b_OLS = np.array(weight_1b_OLS).T

        weight_OLS_mean = np.mean(weight_1b_OLS, axis = 1)
        weight_OLS_std = np.std(weight_1b_OLS, axis = 1)


        # In[17]:


        #print(weight_OLS_std)


        # In[18]:


        
        plt.errorbar(np.arange(100) + 1, weight_OLS_mean.T, yerr=2*weight_OLS_std.T, ecolor = 'r', elinewidth=0.5, capsize=2)
        plt.xlabel("Weight Number")
        plt.ylabel("Value of Weight")
        plt.title("OLS USED MODEL WEIGHTS")
        plt.show()


        # In[19]:




        z = weight_OLS_mean / weight_OLS_std
        p = 2 * (1 - norm.cdf(np.abs(z)))
        ols_sign = np.argwhere(p < 0.05).flatten()

        print("Significant parameters which are extremly different than 0: ")
        print(str(ols_sign))


    # # Question 1 C
        print("Question 1 C")
    # In[20]:


        ITER_NO = 500
        size = 1000
        weight_1c_OLS = []
        np.random.seed(6)
        for i in range(ITER_NO):
            
            random = np.random.choice(np.arange(size), size)
            
            y_b = yn[random]
            x_b = xn[random]
            
            w_holder_c = ridge_regression(x_b, y_b, 339.435353)
            
            weight_1c_OLS.append(w_holder_c)


        weight_1c_OLS = np.array(weight_1b_OLS).T

        weight_OLS_mean_c = np.mean(weight_1c_OLS, axis = 0)
        weight_OLS_std_c = np.std(weight_1c_OLS, axis = 0)


        # In[21]:


        weight_OLS_std_c.shape


        # In[22]:


        weight_1c_OLS.shape


        # In[23]:


        hold = weight_1c_OLS.T
        interval = int(500 * 2.5 /100)
        lower = np.zeros(100)
        higher = np.zeros(100)
        for i in range(100):
            sorted_np = np.sort(hold[i])

            lower[i] = sorted_np[interval]
            higher[i] = sorted_np[500-interval]


        # In[24]:


        
        plt.errorbar(np.arange(100) + 1, weight_OLS_mean_c.T, yerr=2*weight_OLS_std_c.T, ecolor = 'r', elinewidth=0.5, capsize=2)
        plt.xlabel("Weight Number")
        plt.ylabel("Value of Weight")
        plt.title("BEST LAMBDA VALUED RIDGRE REGRESSION USED MODEL WEIGHTS")
        plt.show()


        # In[25]:



        plt.plot(np.arange(100) + 1, weight_OLS_mean_c.T)
        plt.plot(np.arange(100) + 1, lower)
        plt.plot(np.arange(100) + 1, higher)
        plt.xlabel("Weight Number")
        plt.ylabel("Value of Weight")
        plt.title("BEST VALUED LAMBDA USED RIDGE REGRESSION MODEL WEIGHTS")
        plt.legend(["Mean Weigts", "Lower of 95% confidence interval", "Higher of 95% Confidence Interval"])
        plt.show()


        # In[26]:


        z_c = weight_OLS_mean_c / weight_OLS_std_c
        p_c = 2 * (1 - norm.cdf(np.abs(z_c)))
        ols_sign_c = np.argwhere(p_c < 0.05).flatten()

        print("Significant parameters which are extremly different than 0: ")
        print(str(ols_sign_c))


# # QUESTION 2

# # QUESTION 2 A

# In[27]:
    elif question == '2':
        print("Question 2 A")
        with h5py.File('hw3_data3.mat', 'r') as f:
            pop1 = np.array(list(f['pop1']))
            pop2 = np.array(list(f['pop2']))
            vox1 = np.array(list(f['vox1']))
            vox2 = np.array(list(f['vox2']))
            face = np.array(list(f['face']))
            building = np.array(list(f['building']))


        # In[28]:


        print(pop1.shape)
        print(pop2.shape)

        pop_all = np.concatenate((pop1,pop2))


        # In[29]:


        pop_all.shape


        # In[30]:


        def bootstrap_maker(iter_no, size, x, seed):
            
            collector_bootstrap = []
            choice_vector = np.arange(size)
            np.random.seed(seed)
            for i in range(iter_no):
                
                new_places = np.random.choice(choice_vector, size)
                boot_holder = x[new_places]
                collector_bootstrap.append(boot_holder)
                
            np_collector = np.array(collector_bootstrap)
            
            return np_collector


        # In[31]:


        def bootstrap_maker2(iter_no, size, x,seed):
            
            collector_bootstrap = []
            choice_vector = np.arange(size)
            np.random.seed(seed)
            for i in range(iter_no):
                
                new_places = np.random.choice(choice_vector,1)
                boot_holder = x[new_places]
                collector_bootstrap.append(boot_holder)
                
            np_collector = np.array(collector_bootstrap)
            
            return np_collector


        # In[32]:


        samples = bootstrap_maker(1000, 12, pop_all,6)


        # In[33]:


        print(samples.shape)


        # In[34]:


        def mean_dif_finder(pop1, pop2, pop_all, total_size, bins=60):
            
            sample_all = bootstrap_maker(1000, total_size, pop_all,6)
            
            sample_all = sample_all.reshape(1000,total_size)
            
            s1 = sample_all[:,:pop2.size]
            s2 = sample_all[:,pop2.size:]
            
            dif_mean = np.mean(s1, axis=1) - np.mean(s2, axis=1)
            
            prob, values = np.histogram(dif_mean, bins=bins, density=True)
            
            return dif_mean, prob, values


        # In[35]:


        dif_mean, prob, val = mean_dif_finder(pop1, pop2, pop_all, 12)


        # In[36]:


        plt.plot(val[:60], prob)


        # In[37]:


        #print(prob)


        # In[38]:


        
        plt.title("Difference in means of POP1 and POP2 samples")
        plt.xlabel("Difference in Mean (x)")
        plt.ylabel("P(x)")
        plt.bar(val[:60], prob, width=0.1)
        plt.show()


        # In[39]:


        sig = np.std(dif_mean)
        x_head = np.mean(pop1) - np.mean(pop2)
        m_0 = np.mean(dif_mean)

        z = (x_head - m_0) / sig

        p = 2 * (1 - norm.cdf(z))


        # In[40]:


        print('The z value is found as :' + str(z))
        print('The two-tailed p value is found as :' + str(p))


# # Question 2 B

# In[41]:
        print("Question 2 B")

        def conf_int_calculator(arr, level):
            size = arr.shape[0]
            
            interval = (100 - level)/2
            
            per_level = int(size * (interval / 100))
            print(per_level)
            lower_conf = arr[per_level]
            higher_conf = arr[size - per_level]
            
            return lower_conf, higher_conf


        # In[42]:


        print(vox1.shape)
        print(vox2.shape)


        # In[43]:


        vox1_b = bootstrap_maker(1000, 50, vox1,7)
        vox2_b = bootstrap_maker(1000, 50, vox2,57)


        # In[44]:


        #vox1_b = vox1_b.reshape(1000,1)
        #vox2_b = vox2_b.reshape(1000,1)


        # In[45]:


        collector_boot = np.zeros(1000)
        for i in range(1000):
            collector_boot[i] = np.corrcoef(vox1_b[i].flatten(), vox2_b[i].flatten())[0,1]


        # In[46]:


        #print(collector_boot.shape)
        mean_corr = np.mean(collector_boot)
        collector_boot = np.sort(collector_boot, axis=None)
        lower_conf, higher_conf = conf_int_calculator(collector_boot, 95)


        # In[47]:


        print("Mean of the correlation matrix vector values is: " + str(mean_corr))
        print("Lower confidence value in 95% confidence interval is: " + str(lower_conf))
        print("Higher confidence value in 95% confidence interval is: " + str(higher_conf))


        # In[48]:


        zero_precentile = 100*(np.size(np.where(collector_boot == 0))/1000)


        # In[49]:


        print("Zero percantage correlation value is : " + str(zero_precentile))


# # Question 2 C

# In[50]:
        print("Question 2 C")

        vox1_c = bootstrap_maker(1000, 50, vox1, 5)
        vox2_c = bootstrap_maker(1000, 50, vox2, 17)


        # In[51]:


        collector_boot_c = np.zeros(1000)
        for i in range(1000):
            collector_boot_c[i] = np.corrcoef(vox1_c[i].flatten(), vox2_c[i].flatten())[0, 1]
            


        # In[52]:


        #print(collector_boot_c)


        # In[53]:


        boot_c_overline = np.corrcoef(vox1.flatten(), vox2.flatten())[0 , 1]
        sig_c = np.std(collector_boot_c)
        mean_0 = np.mean(collector_boot_c)

        z = (boot_c_overline - mean_0)/sig_c
        p = 1 - norm.cdf(z)


        # In[54]:


        print("Z value found is: " + str(z))
        print("P value found is: " + str(p))


        # In[55]:


        prob, values = np.histogram(collector_boot_c, bins=60, density=True)


        # In[56]:


        
        plt.title("Correlation between Vox1 and Vox2")
        plt.xlabel("Correleation c")
        plt.ylabel("P(c)")
        plt.bar(val[:60], prob, width=0.1)
        plt.show()


# # Question 2 D

# In[57]:

        print("Question 2 D")
        print(face.shape)
        print(building.shape)


        # In[58]:


        face_b = bootstrap_maker(1000, 20, face, 6)
        build_b = bootstrap_maker(1000, 20, building, 6)


        # In[59]:


        def dif_mean_2d(arr1, arr2, seed, iter_no):
            
            np.random.seed(seed)
            collector_dif = []
            
            for i in range(iter_no):
                collector = []
                for j in range(arr1.shape[0]):
                    opt = np.zeros((1,2))
                    opt[0,0] = (arr1[j] - arr2[j])
                    opt[0,1] = (arr2[j] - arr1[j])
                    collector.append(np.random.choice(opt.flatten()))
                collector_dif.append(np.mean(collector))
            collector_dif = np.array(collector_dif)
            
            return collector_dif


        # In[60]:


        dif_mean_d = dif_mean_2d(face, building, 6, 1000)


        # In[61]:


        prob_d, values_d = np.histogram(dif_mean_d, bins=60, density=True)


        # In[62]:


        
        plt.title("Difference in mean of the samples assuming they are same")
        plt.xlabel("Difference in mean (x)")
        plt.ylabel("P(x)")
        plt.bar(values_d[:60], prob_d, width=0.1)
        plt.show()


        # In[63]:


        x_o = np.mean(build_b) - np.mean(face_b)
        sig_d = np.std(dif_mean_d)
        mean_0_d = np.mean(dif_mean_d)

        z_d = (x_o - mean_0_d)/sig_d

        p_d = 2 * (1 - norm.cdf(np.abs(z)))


        # In[64]:


        print("Z value found is: " + str(z_d))
        print("Two-sided P value found is: " + str(p_d))


# # Question 2 E
        print("Question 2 E")
# In[65]:


        face_build = np.concatenate((face,building), axis=0)


        # In[66]:


        diff_mean_e, prob_e, vals_e = mean_dif_finder(face, building, face_build, 40, bins=60)


        # In[67]:


        
        plt.title("Difference in mean (x) vs. P(x) without any assumptions")
        plt.xlabel("Difference in mean (x)")
        plt.ylabel("P(x)")
        plt.bar(vals_e[:60], prob_e, width=0.1)
        plt.show()


        # In[68]:


        x_o = np.mean(building) - np.mean(face)
        sig_e = np.std(diff_mean_e)
        mean_0_e = np.mean(diff_mean_e)

        z_e = (x_o - mean_0_e)/sig_e

        p_e = 2 * (1 - norm.cdf(np.abs(z)))


        # In[69]:


        print("Z value found is: " + str(z_e))
        print("Two-sided P value found is: " + str(p_e))
        
    else:
        print("Invalid number for quesion has entered. Please try again, make sure you have enetered a number which is a real question number...")


# In[ ]:

alp_kumbasar_21602607_HW3(question)


