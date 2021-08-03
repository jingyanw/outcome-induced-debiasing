import numpy as np
import os
import time

from estimator import *

import matplotlib
matplotlib.use('TKAgg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Computer Modern'
import matplotlib.pyplot as plt

LAMBDAS = [0, 1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1,\
				2, 4, 8, 16, 32, np.inf]
NUM_SAMPLES = 100 # number of sampled total orderings to approximate interpolation

fontsize=25
legendsize=20
ticksize=17.5
linewidth=2.5
markersize=10
markeredgewidth=4
axissize=17.5

PLOT_DIR = 'plots'
MARKERS = {
	'mean': 'x',
	'median': 'o',
	'subsample': 's',
	'cv': '<',
	'oracle': 'd',
}
LINESTYLES = {
	'mean': 'dotted',
	'median': 'dashed',
	'subsample': 'dashdot',
	'cv': 'solid',
	'oracle': (0, (1, 5)),
}
COLORS = {
	'mean': '#ff7f0e', # orange
	'median': '#d62728', # red
	'subsample': '#9467bd', # purple
	'cv': '#1f77b4', # blue
	'oracle': '#7f7f7f', # gray
}

text_l2_err = r'squared $\ell_2$ error'
LAMBDAS_TEXT = [str(lda) for lda in LAMBDAS]
LAMBDAS_TEXT[1:9] = ['\\frac{1}{512}', '\\frac{1}{256}', '1/128', '\\frac{1}{64}', '1/32', '\\frac{1}{16}', '1/8', '\\frac{1}{4}']
LAMBDAS_TEXT[-1] = '\infty'

# Run simulation varying the paramter N
# MODE_ORDER: noninterleaving | interleaving | binary
# MODE_BIAS_NOISE: bias | noise | noisy
def simulate_vary_n(mode_order='noninterleaving', mode_bias_noise='bias', lambdas=LAMBDAS, num_samples=NUM_SAMPLES):
	(delta, epsilon) = map_mode_bias_noise(mode_bias_noise)

	if mode_order in ['noninterleaving', 'interleaving']:
		d = 3
		ns = [2, 5, 10, 25, 50, 100]
	elif mode_order =='binary' : # binary (10%/90%)
		d = 4
		ns = [10, 20, 30, 40, 50, 70, 100]
	else: raise Exception('Unknown mode order')
	print('simulate_vary_n [d=%d | order %s | bias-noise %s]... ' % (d, mode_order, mode_bias_noise))

	repeat = 250

	ns = np.array(ns, dtype=int)
	N = len(ns)
	L = len(lambdas)

	ests_cv_sample = np.zeros((d, N, repeat))
	ests_mean = np.zeros((d, N, repeat))
	ests_median = np.zeros((d, N, repeat))
	ests_subsample_weighted = np.zeros((d, N, repeat))
	ests_solution = np.zeros((d, N, L, repeat))

	# \lambda count
	counts_il = np.zeros((N, L))

	tic = time.time()
	for i in range(N):
		n = ns[i]

		if mode_order == 'noninterleaving':
			ordering = np.reshape(np.arange(d*n), (d, n))
		elif mode_order == 'interleaving':
			ordering = np.reshape(np.arange(d*n), (d, n), order='F')
		elif mode_order =='binary':
			n_percent = 0.1
			m = int(n_percent * n)

			assert(d % 2 == 0)
			d_half = int(d/2)
			ordering = np.zeros((d, n), dtype=int)
			ordering[:d_half, (n-m):] = 1
			ordering[d_half:, m:] = 1

		print('ordering:')
		print(ordering)

		print('[n: %d/%d] %d iters...' % (i+1, N, repeat))
		for r in range(repeat):
			if r % 10 == 0:
				print('%d/%d (%s | %d sec)' % (r+1, repeat, time.strftime("%H:%M", time.localtime()), time.time() - tic))

			# normal
			noise = np.random.normal(size=(d, n)) * epsilon
			bias = generate_bias_marginal_gaussian(d, n, ordering) * delta
			y = noise + bias

			(il_sample, xs_sample) = \
				perform_cv(y, lambdas, ordering, num_samples=num_samples)

			# compute solutions for all LAMBDAS
			# XS: d x L
			# BS: d x n x L
			(xs, bs) = solve_opt(y, lambdas, ordering=ordering)		

			ests_cv_sample[:, i, r] = xs[:, il_sample]

			ests_mean[:, i, r] = np.mean(y, axis=1)
			ests_median[:, i, r] = np.median(y, axis=1)
			ests_subsample_weighted[:, i, r] = solve_subsampling_weighted_recenter(y, ordering)
			ests_solution[:, i, :, r] = xs # d x n x L x repeat

			counts_il[i, il_sample] += 1

	# FIG 1: L2 err vs. n
	errs_cv_sample_l2 = l2(ests_cv_sample)

	errs_mean_l2 = l2(ests_mean)
	errs_median_l2 = l2(ests_median)
	errs_subsample_weighted_l2 = l2(ests_subsample_weighted)

	errs_bestfixed_l2 = keep_best_fixed(ests_solution)

	fig = plt.figure()
	ax = plt.subplot(111)

	ax.tick_params(axis='x', labelsize=ticksize)
	ax.tick_params(axis='y', labelsize=ticksize)
	ax.tick_params(axis='x', which='minor', bottom=False)

	ax.errorbar(ns, np.mean(errs_mean_l2, axis=1), np.std(errs_mean_l2, axis=1) / np.sqrt(repeat),
		label='mean', color=COLORS['mean'], marker=MARKERS['mean'], linestyle=LINESTYLES['mean'],
		markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)

	ax.errorbar(ns, np.mean(errs_median_l2, axis=1), np.std(errs_median_l2, axis=1) / np.sqrt(repeat),
		label='median', color=COLORS['median'], marker=MARKERS['median'], linestyle=LINESTYLES['median'],
		markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)

	if mode_order == 'binary': # no subsample for total order

		ax.errorbar(ns, np.mean(errs_subsample_weighted_l2, axis=1), yerr=np.std(errs_subsample_weighted_l2, axis=1) / np.sqrt(repeat), 
			label='weighted mean', color=COLORS['subsample'], marker=MARKERS['subsample'], linestyle=LINESTYLES['subsample'],
			markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)

	ax.errorbar(ns, np.mean(errs_cv_sample_l2, axis=1), yerr=np.std(errs_cv_sample_l2, axis=1) / np.sqrt(repeat),
		label='CV', color=COLORS['cv'], marker=MARKERS['cv'], linestyle=LINESTYLES['cv'],
		markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)

	ax.errorbar(ns, np.mean(errs_bestfixed_l2, axis=1), yerr=np.std(errs_bestfixed_l2, axis=1)/ np.sqrt(repeat),
		label='best fixed ' + r'$\lambda$', color=COLORS['oracle'], marker=MARKERS['oracle'], linestyle=LINESTYLES['oracle'],
		markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)

	plt.xlabel(r'$n$', fontsize=axissize)
	if mode_bias_noise == 'bias':
		plt.ylabel(text_l2_err, fontsize=axissize)

	plt.xscale('log')
	plt.yscale('log')

	if mode_order == 'binary':
		ax.set_xticks([10, 20, 30, 40, 50, 70, 100])
		ax.set_xticklabels([r'$10$', r'$20$', r'$30$', r'$40$', r'$50$', r'$70$', r'$100$'])
	else:
		ax.set_xticks([2, 5, 10, 25, 50,100])
		ax.set_xticklabels([r'$2$', r'$5$', r'$10$', r'$25$', r'$50$', r'$100$'])

	ax.tick_params(axis='x', labelsize=ticksize)
	ax.tick_params(axis='y', labelsize=ticksize)

	plt.savefig('%s/vary_n_%s_%s_d%d.pdf' % (PLOT_DIR, mode_order, mode_bias_noise, d),
				bbox_inches='tight')

	# count \lambda
	if mode_order == 'binary':
		# only plot d == 50
		i = np.where(ns == 50)[0][0]
		fig = plt.figure()
		ax = plt.subplot(111)
		xs = np.arange(L)
		width = 0.7

		probs = counts_il[i, :] / repeat
		ax.bar(xs , probs, width, label='CV', color=COLORS['cv'])

		plt.xlabel(r'$\lambda$', fontsize=axissize)
		if mode_bias_noise == 'bias':
			plt.ylabel('Fraction of times', fontsize=axissize)
		ax.set_xticks(xs)

		text_lda = [r'$%s$' % lda for lda in LAMBDAS_TEXT]
		for il in np.arange(1, L, 2):
			text_lda[il] = ''
		ax.set_xticklabels(text_lda)

		ax.tick_params(axis='x', labelsize=ticksize)
		ax.tick_params(axis='y', labelsize=ticksize)

		plt.savefig('%s/vary_n_lambda_%s_%s_d%d.pdf' % (PLOT_DIR, mode_order, mode_bias_noise, d),
					bbox_inches='tight')

# MODE: bias | mixed | noise
def map_mode_bias_noise(mode):
	if mode == 'bias':
		return (1, 0)
	elif mode == 'mixed':
		return (0.5, 0.5)
	elif mode == 'noise':
		return (0, 1)

# Generate marginal-Gaussian bias obeying ORDERING
# ORDERING: d-by-n
def generate_bias_marginal_gaussian(d, n, ordering=None):
	idxs = sample_total_from_partial(d, n, ordering=ordering) # increasing order

	bias = np.zeros(d*n)
	bias[idxs] = np.sort(np.random.normal(size=d*n))
	bias = np.reshape(bias, (d, n))
	return bias

# X: d * T1 * T2 * ...
# # X_TRUE: length-D vector
# Compute L2 along axis 0
# # Returns:
# # 	ERR: T1 * T2 * ...
def l2(x, x_true=None):
	if x_true is None:
		x_true = np.zeros(x.shape[0])
	else:
		assert(x_true.ndim == 1)
		assert(x.shape[0] == len(x_true))

	n = x.ndim
	for _ in range(n-1):
		x_true = np.expand_dims(x_true, axis=1)

	err = np.square(x - x_true)
	err = np.mean(err, axis=0)
	
	return err

# Y: d-by-n
# ORDERING can be group grades (with an inherent ordering, or simply types (without an ordering on the types)
# deterministic (no randomness)
def solve_subsampling_weighted(Y, ordering, mask_data=None):
	(d, n) = Y.shape
	if mask_data is None:
		mask_data = np.full((d, n), True)

	assert(is_valid_ordering(ordering))

	T = int(np.max(ordering)) + 1 # number of types (index from 0)
	assert(np.min(ordering) == 0)

	counts = np.zeros((d, T), dtype=int)
	for t in range(T):
		counts[:, t] = np.sum((ordering == t) * mask_data, axis=1)

	counts = np.min(counts, axis=0) # len-T array

	C = np.sum(counts)
	if C == 0:
		return np.full(d, np.nan)

	ests = np.zeros(d)
	for i in range(d):
		for t in range(T):
			if counts[t] == 0: continue
			idxs = np.where(np.logical_and(ordering[i, :] == t, mask_data[i, :]) )[0]
			assert(len(idxs) >= counts[t])
			ests[i] += np.mean(Y[i, idxs]) * counts[t] / C

	return ests

def solve_subsampling_weighted_recenter(Y, ordering, mask_data=None):
	if mask_data is None:
		(d, n) = Y.shape
		mask_data = np.full((d, n), True)

	xs = solve_subsampling_weighted(Y, ordering, mask_data=mask_data)
	counts = np.sum(mask_data, axis=1)
	shift = np.sum(Y * mask_data) / np.sum(mask_data) - np.sum(counts * xs) /  np.sum(counts)

	xs =  xs + shift
	return xs

# Y: d x T x L x repeat or d x L x repeat (T: #settings)
# Returns: 
# 	ERRS_BEST_FIXED: T x repeat matrix or repeat-array
def keep_best_fixed(y):
	if len(y.shape) == 4:
		T = y.shape[1]
		repeat = y.shape[-1]

		errs_best_fixed = np.zeros((T, repeat))
		for t in range(T):
			errs_best_fixed[t, :] = keep_best_fixed(y[:, t, :, :])
		return errs_best_fixed

	elif len(y.shape) == 3:
		err = l2(y) # L x repeat
		err_mean = np.mean(err, axis=1) # length-L
		il = np.argmin(err_mean)
		return err[il, :] # length-repeat

######
if __name__ == '__main__':
	np.random.seed(0)

	if not os.path.exists(PLOT_DIR):
		os.makedirs(PLOT_DIR)

	for mode_order in ['noninterleaving', 'interleaving', 'binary']:
		for mode_bias_noise in ['bias', 'mixed', 'noise']:
			simulate_vary_n(mode_order=mode_order, mode_bias_noise=mode_bias_noise)
