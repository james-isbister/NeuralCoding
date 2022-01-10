import numpy as np
import matplotlib.pyplot as plt
import tqdm as tq
from scipy.stats import gaussian_kde, beta
from scipy.signal import argrelmax


def gaussian(x, mu=0, sigma=1):
    return ((2*np.pi*sigma**2)**-0.5) * np.exp(-0.5 * ((x - mu)/sigma)**2)

def gaussian_deriv(x, mu=0, sigma=1):
    return -((x-mu)/sigma**2)*gaussian(x, mu, sigma)

def gaussian_deriv_2(x, mu=0, sigma=1):
    return ((x-mu)**2/sigma**2 - 1)*gaussian(x, mu, sigma)/sigma**2

def gaussian_kde_deriv(x, X, sigma):
    if np.isscalar(x):
        x = np.array([x])
    return gaussian_deriv(x.repeat(len(X)).reshape(len(x), len(X)), mu=X, sigma=sigma).mean(axis=1)

def gaussian_kde_deriv_2(x, X, sigma):
    if np.isscalar(x):
        x = np.array([x])
    return gaussian_deriv_2(x.repeat(len(X)).reshape(len(x), len(X)), mu=X, sigma=sigma).mean(axis=1)

def count_kde_modes(K, qk):
    modes = find_all_kde_modes(K, qk)
    return len(modes)

def find_all_kde_modes_old(K, qk):    
    f   = lambda x : -K.pdf(x)
    df  = lambda x : -gaussian_kde_deriv(x,   X=K.dataset.squeeze(), sigma=K.factor * K.dataset.std(ddof=1))
    ddf = lambda x : -gaussian_kde_deriv_2(x, X=K.dataset.squeeze(), sigma=K.factor * K.dataset.std(ddof=1))
    
    Xmin = K.dataset.min()
    Xmax = K.dataset.max()
    Xstd  = K.dataset.std()
    
    return find_all_minima(f=f, df=df, ddf=ddf, low=Xmin - Xstd/4, high=Xmax + Xstd/4, break_after_n=qk)

def find_all_minima_old(f, df, ddf, random_starts=500, low=-100.0, high=100, break_after_n=False):
    minima = np.array([])

    for ix in range(random_starts):
        x0 = np.random.uniform(low=low, high=high)
        results = opt.root(df, x0=x0, jac=ddf)
#         new_minimum = opt.minimize(f, x0=x0, jac=df, hess=ddf, method='Newton-CG', options={'xtol': 1e-3})['x']
        if ddf(results['x']) > 0 and results['success']:
            if np.any(minima):
                if np.any([np.isclose(results['x'], minimum, rtol=1e-2) for minimum in minima]):
                    pass
                else:
                    minima = np.concatenate((minima, results['x']))
            else:
                minima= np.concatenate((minima, results['x']))
                
        if break_after_n:
            if len(minima) > break_after_n:
                break
    minima.sort()
#     minima = minima[ddf(minima) > 1e-12]
#     minima = minima[np.abs(df(minima)) < 1e-6 ]
    return minima

def find_all_kde_modes_smart(K, break_after_n):
    f      = lambda x : K.pdf(x)
    negf   = lambda x : -f(x)

    df     = lambda x : gaussian_kde_deriv(x, K.dataset[0], K.factor * K.dataset.std(ddof=1))
    negdf  = lambda x : -df(x)

    ddf    = lambda x : gaussian_kde_deriv_2(x, K.dataset[0], K.factor * K.dataset.std(ddof=1))
    negddf = lambda x : -ddf(x)

    #------------------

    b_left  = K.dataset.min() - K.dataset.std()/4
    b_right = K.dataset.max() + K.dataset.std()/4

    x_init  = K.dataset.min()

    ftol=1e-32
    eps=1e-12
    gtol=1e-12

    j_maxs = []
    f_maxs = []
    j_mins = []
    f_mins = []
    
    #--------------------
    
    while not np.isclose(b_left, b_right, 1e-1):
        for fun, jac, mins in zip([negdf, negf, df, f], [negddf, negdf, ddf, df], [j_maxs, f_maxs, j_mins, f_mins]):
            results_lbfgsb_jac = opt.minimize(fun, x0=x_init, bounds=([b_left, b_right],), jac=jac, options={'ftol' : ftol, 'gtol' : gtol}, method='L-BFGS-B')
            results_slsqp_jac  = opt.minimize(fun, x0=x_init, bounds=([b_left, b_right],), jac=jac, options={'ftol' : ftol}, method='SLSQP')

            results_lbfgsb = opt.minimize(fun, x0=x_init, bounds=([b_left, b_right],), options={'ftol' : ftol, 'eps' : eps, 'gtol' : gtol}, method='L-BFGS-B')
            results_slsqp  = opt.minimize(fun, x0=x_init, bounds=([b_left, b_right],), options={'ftol' : ftol, 'eps' : eps}, method='SLSQP')

            results_nm     = opt.minimize(fun, x0=x_init, method='Nelder-Mead', options={'fatol' : 1e-16, 'initial_simplex' : np.array([b_left,x_init]).reshape(-1,1)})
            if results_nm['x'] < x_init:
                results_nm['success'] = False

            fun_min = min([results['x'] for results in [results_lbfgsb_jac, results_slsqp_jac, results_lbfgsb, results_slsqp, results_nm] if results['success']])

            for eta in [0.1, 0.25, 0.5, 0.75, 0.9, 0.925, 0.95, 0.99, 0.999]:
                results_tnc_jac = opt.minimize(fun, x0=x_init, bounds=([b_left, b_right],), jac=jac, options={'ftol' : ftol, 'gtol' : gtol,  'eta' : eta}, method='tnc')
                results_tnc    = opt.minimize(fun, x0=x_init, bounds=([b_left, b_right],), options={'ftol' : ftol, 'gtol' : gtol, 'eps' : eps, 'eta' : eta}, method='tnc')
                try:
                    tnc_min = min([results['x'] for results in [results_tnc_jac, results_tnc] if results['success']])
                except ValueError:
                    pass

            fun_min = min(fun_min, tnc_min)

            b_left = x_init
            x_init = fun_min
            mins.append(fun_min[0])

            while f(x_init) < 1e-6 and x_init < b_right:
                x_init+=1e-3
        
            while np.abs(df(x_init)) < 1e-5 and np.abs(ddf(x_init)) < 1e-2:
                x_init+=1e-3

            if np.isclose(b_left, b_right, 1e-1):
                break
                
    f_maxs = np.array(f_maxs)
    f_maxs = f_maxs[ddf(f_maxs) < 0]
    return np.array(f_maxs)

def find_all_kde_modes(K, dx=0.001):
    '''
    find_all_kde_modes(K, dx=0.001)
    
    Find all modes of kernel density estimate via exhaustive search.
    No 'smarter' method could guarantee finding every mode. Make dx
    small enough that you can detect new modes at high enough
    bandwidths, but not so small that the search takes a long time.
    
    Arguments:
        - K (scipy.stats.kde.gaussian_kde): Gaussian KDE object
        - dx (float) : search increment (default= 0.001)
        
    Return:
        - modes (np.array) : modes of kernel density estimate
    '''
    f = lambda x : K.pdf(x)
    
    # Grid search
    x_min  = K.dataset.min() - K.dataset.std()/4
    x_max = K.dataset.max() + K.dataset.std()/4
    x = np.arange(x_min, x_max, dx)
    modes = x[argrelmax(f(x))[0]]
    
    return modes

def count_kde_modes(K, dx=0.001):
    '''
    count_kde_modes(K, dx=0.001)
    
    Count number of modes in kernel density estimate
    
    Arguments:
        - K (scipy.stats.kde.gaussian_kde): Gaussian KDE object
        - dx (float): search increment

    Return:
        - count (int) : number of modes in kernel density estimate
    '''
    return len(find_all_kde_modes(K, dx=0.001))

def find_critical_bandwidth(K, num_modes=1, tol=1e-16, min_sigma = 1e-3, max_sigma = 1e2):
    '''
    find_critical_bandwidth(K, qk=1, tol=1e-16, min_sigma = 1e-3, max_sigma = 1e2)
    
    Find critical bandwidth via binary search.
    
    Arguments:
        - K (scipy.stats.kde.gaussian_kde): Gaussian KDE object
        - num_modes (int): number of modes to find critical bandwidth for
        - tol (float) : tolerance in critical bandwidth search (default=1e-16)
        - min_sigma (float) : initial lower bound on search
        - max_sigma (float) : initial upper bound on search
    
    Returns:
        - sigma_critical (float) : critical bandwidth
    '''
    
    return binary_search(K, num_modes, tol= tol, min_sigma= min_sigma, max_sigma= max_sigma)

def binary_search(K, target_modes, tol= 1e-16, min_sigma= 1e-3, max_sigma= 1e2):
    '''
    binary_search(K, num_modes, tol= 1e-16, min_sigma= 1e-3, max_sigma= 1e2)
    
    Binary search algorithm. 
        1. Initialise minimum and maximum bandwidth
        2. Set test bandwidth as midpoint (or log_10 mid point) between max and min
        3. Count number of modes in KDE with this bandwidth
        4. If KDE has more modes than target number, set minimum
           bandwidth to test bandwidth, else set maximum bandwidth
           to test bandwidth
        5. Repeat steps 2-4 until max - min is less than tol
        6. Return maximum bandwidth
        
    
    Arguments:
        - K (scipy.stats.kde.gaussian_kde): Gaussian KDE object
        - target_modes (int): target number of modes to find critical bandwidth for
        - tol (float) : tolerance in critical bandwidth search (default=1e-16)
        - min_sigma (float) : initial lower bound on search
        - max_sigma (float) : initial upper bound on search
    
    Returns:
        - max_sigma
    
    '''
    
    while not np.isclose(max_sigma - min_sigma, tol):
        sigma = np.round(10**(0.5*(np.log10(min_sigma) + np.log10(max_sigma))), int(-np.log10(tol)))
        K.set_bandwidth(sigma/K.dataset.std(ddof=1))
        num_modes = count_kde_modes(K, dx=1e-3);
        if num_modes > target_modes:
            min_sigma = sigma
        elif num_modes <= target_modes:
            max_sigma = sigma
    
    return max_sigma
            

def bootstrap_sample(X, sigma, sample_size):
    '''
    bootstrap_sample(X, sigma, sample_size)
    
    Generate smoothed bootstrapped sample from kernel density estimate.
    
    Arguments:
        - X : original data
        - sigma : bandwidth of kernel density estimate for smoothing datapoints
        - sample_size : size of bootstrapped sample
        
    Returns:
        - Z : bootstrapped sample data
    '''
    
    return (1 + sigma ** 2 / X.var(ddof=0))**-0.5 * (np.random.choice(X, size=sample_size, replace=True) + np.random.randn(sample_size)*sigma)

def silvermans_test(X, target_modes=1, num_bootstraps = 200, tol=1e-3):
    '''
    silvermans_test(X, num_modes=1, num_bootstraps = 200, tol=1e-3)
    
    Estimate p-value for  Silverman's hypothesis test that the true probability 
    density generating data X has more than k modes.
    
    1. Construct kernel density estimate of data X
    2. Find critical bandwidth of kernel density estimate for k modes
    
    For N repeats:
        3. Generate smoothed bootstrap sample Z of data X using critical bandwidth
        4. Construct kernel density estimate of Z using critical bandwidth
    
    5. Estimate p-value = Number of times bootstrapped KDE had less than or equal
       to number of target modes / N
       
   Arguments:
        - X (np.array) : data to construct KDE
        - target_modes (int) : number of modes to test
        - num_bootstraps (int) : number of bootstrap samples to estimate p-value
        - tol (float) : tolerance on critical bandwidth estimate
        
    Returns:
        - p (float) : p-value of hypothesis test
        
    '''
    
    # Create Gaussian KDE object
    K = gaussian_kde(X)
    
    # find critical bandwidth
    sigma_critical = find_critical_bandwidth(K, num_modes=target_modes, tol=tol)
    
    # Set KDE kernel bandwidth to critical
    K.set_bandwidth(sigma_critical/K.dataset.std(ddof=1))
    
    # Estimate p-value
    p = 0
    
    for ix in tq.tqdm(list(range(num_bootstraps))):
        # Sample from KDE with critical bandwidth
        Z = bootstrap_sample(X, sigma=sigma_critical, sample_size=len(X))

        # Create KDE object for new sample
        Kz = gaussian_kde(Z)
        Kz.set_bandwidth(sigma_critical/Kz.dataset.std(ddof=1))

        num_modes = count_kde_modes(Kz, dx=1e-3)
        p+= float(target_modes>=num_modes)/num_bootstraps
    
    return p

def silvermans_test_beta(X, target_modes=1, tol=1e-3, p_significant = 0.9, significance_credibility=0.995, CI_width=None, max_num_bootstraps=10000, min_num_bootstraps=5):
    '''
    silvermans_test_beta(X, num_modes=1, tol=1e-3, p_significant = 0.9, significance_credibility=0.995, CI_width=None, max_num_bootstraps=10000)
    
    Variant of silverman's test where we don't specify number of bootstrapped
    samples we  will take to estimate p-value. Instead we model mode count
    comparison as a bernoulli random variable and infer p-value as the MAP
    estimate of a beta distribution. If we only care about whether test is
    significant, we can terminate test when we reach a given (in)credibility value
    for the significance threshold p-value.
    
    If we want to get a credible estimate of the p_value, set a threshold on the
    variance of the beta distribution to terminate at.
    
    For safety, set an upper limit on number of bootstraps
    
    Arguments:
        - X (np.array) : data to construct kernel density estimate
        - target_modes (int) : number of modes to test
        - p_significant (float) : significance value
        - significance_credibility (float) : credibility of acceptance/rejection
        - CI_width (float) : acceptable credibility of p-value to terminate estimation
        - max_num_bootstraps (int) : maximum number of repeats
        
    Returns:
        - p (float) : p-value of hypothesis test
    '''
    
    K = gaussian_kde(X)
    
    # find critical bandwidth
    sigma_critical = find_critical_bandwidth(K, num_modes=target_modes, tol=tol)
    
    # Set KDE kernel bandwidth to critical
    K.set_bandwidth(sigma_critical/K.dataset.std(ddof=1))
    
    # Estimate p-value
    
    a = 1
    b = 1
    repeats = 0
    while True:
        Z = bootstrap_sample(X, sigma=sigma_critical, sample_size=len(X))
        Kz = gaussian_kde(Z)
        Kz.set_bandwidth(sigma_critical/Kz.dataset.std(ddof=1))
        num_modes = count_kde_modes(Kz)
        if target_modes >= num_modes:
            a += 1
        else:
            b += 1
        
        repeats += 1
        
        if repeats >= min_num_bootstraps:
    
            if repeats >= max_num_bootstraps:
                break
            
            elif (beta.cdf(p_significant, a, b) < (1 - significance_credibility)) or (beta.cdf(p_significant, a, b) > significance_credibility):
                if CI_width is not None:
                    if np.diff(beta.interval(0.95, a, b)) < (CI_width):
                        break
                    else:
                        continue
                else:
                    break
                
#     print('Terminated after %i samples. CI = %.4f'%(repeats, np.diff(beta.interval(0.95, a, b))))
    return (a - 1.)/(a + b - 2.)

def plot_kde(K):
    '''
    Visualise kernel density estimate
    
    Arguments:
        - K (scipy.stats.kde.gaussian_kde) : gaussian kernel density estimate
    
    '''
    
    Xmin = K.dataset.min()
    Xmax = K.dataset.max()
    Xstd  = K.dataset.std(ddof=1)
    x = np.linspace(Xmin-Xstd/4, Xmax+Xstd/4, 1000)
    plt.plot(x, K.pdf(x), lw=4)
    
def plot_modes(K, modes):
    '''
    Plot modes of KDE
    '''
    
    plt.plot(modes, K.pdf(modes), 'o', ms=10)