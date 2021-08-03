# Implement our estimator and cross-validation
import numpy as np
import cvxpy as cp

# Split data into training-validation sets
def split_trainval_random_pair(d, n, ordering=None):
	idxs = sample_total_from_partial(d, n, ordering=ordering)
	
	train = np.full((d, n), False)
	val = np.full((d, n), False)

	for i in range(d): # for each row
		idx_first = n * i
		idx_last = n * (i+1) - 1
		idxs_row = list(filter(lambda i: i >= idx_first and i <= idx_last, idxs)) # preserve ordering

		L = int(len(idxs_row) / 2)
		for t in range(L):
			if np.random.uniform() < 0.5:
				(idx_train, idx_val) = idxs_row[2*t : 2 * (t+1)]
			else:
				(idx_val, idx_train) = idxs_row[2*t : 2 * (t+1)]

			train[i, idx_train % n] = True
			val[i, idx_val % n] = True

		if L * 2 < len(idxs_row): # odd
			assert(len(idxs_row) - L*2 == 1)
			# assign last element to val
			val[i, idxs_row[-1] % n] = True

	assert(not np.any(np.logical_and(train, val)) )
	assert(np.array_equal(np.logical_or(train, val), np.full((d, n), True)))
	return train, val

# Perform CV
# Returns:
# 	IL_CV: scalar index
# 	XS_CV: d-array
def perform_cv(y, lambdas, ordering=None, num_samples=100):
	(d, n) = np.shape(y)
	(set_train, set_val) = split_trainval_random_pair(d, n, ordering)

	# train
	# XS_TRAIN: (d, L)
	# BS_TRAIN: (d, n, L)  (only the train entries have non-zero values)
	(xs_train, bs_train) = solve_opt(y, lambdas, mask=set_train, ordering=ordering)

	L = len(lambdas)

	# compute val error
	errs_val = np.zeros(L)

	for il in range(L):
		xs = xs_train[:, il]

		bs_val = interpolate_values(bs_train[:, :, il], set_train, set_val, \
						ordering=ordering, num_samples=num_samples)
		y_interpolate = bs_val + xs_train[:, il][:, np.newaxis]
		y_interpolate[np.logical_not(set_val)] = 0 # just to be safe
		errs_val[il] = np.mean(np.square( (y_interpolate - y) * set_val ))

	# choose \lambda (CV)
	xs_cv = np.zeros(d)

	err_min = np.min(errs_val)
	il_cv = np.where(errs_val == err_min)[0]
	if len(il_cv) > 1:
		print('Multiple choices of optimal lambda in CV!')
	il_cv = np.random.choice(il_cv) # breaking ties
	xs_cv = xs_train[:, il_cv]

	return il_cv, xs_cv

# Interpolate values in SET_VAL -- take neighbor wrt ORDERING
# SET_TRAIN/VAL: dxn mask {True, False}
# NUM_SAMPLE: number of sampled total orderings
# ORDERING: take the nearest neighbor according to TOTAL_ORDER
# Return: 
# 	BS: filled in SET_VAL
def interpolate_values(bs, set_train, set_val, ordering=None, num_samples=100):
	(d, n) = bs.shape

	bs[set_val] = 0 # just to be safe
	idxs_train = np.where(set_train.flatten())[0]

	for r in range(num_samples):
		total_order = sample_total_from_partial(d, n, ordering=ordering)
		positions_train = np.isin(total_order, idxs_train)
		positions_train = np.where(positions_train)[0] # positions of train samples in TOTAL_ORDER

		for i in range(d):
			for j in range(n):
				if not set_val[i, j]:
					continue
				idx_val = np.ravel_multi_index((i, j), (d, n))
				position_val = np.where(total_order == idx_val)[0][0]

				diff = np.abs(positions_train - position_val)
				diff_min = np.min(diff)

				positions_neighbor = positions_train[diff == diff_min]
				assert(len(positions_neighbor) <= 2)
				idxs_neighbor = total_order[positions_neighbor] # positions_neighbor is a set of size 1 or 2, not a single element
				for idx_neighbor in idxs_neighbor:
					assert(idx_neighbor in idxs_train)

					(i_neighbor, j_neighbor) = np.unravel_index(idx_neighbor, (d, n))
					bs[i, j] += bs[i_neighbor, j_neighbor] / len(idxs_neighbor) / num_samples

	return bs

# Inputs:
# Y: d-by-n matrix -- rating data
# 	D: number of courses
#	N: number of ratings
# LAMBDAS: an L-array of candidate lambda values
# 	Can handle inf (naive sample mean)
# MASK: d x n matrix
# 	True: use this score for estimation (train) | False: don't use this score for estimation (save for val)
# ORDERINGS (of grades): d-by-n
# 	e.g. [1, 0, 2] means b1 < b0 < b2
# MODE_FAST: use properties of the optimization to derive constraints
# 	Does not change the results

# Returns:
# 	XS: d x L
# 	BS: d x n x L
def solve_opt(Y, lambdas, mask=None, ordering=None, mode_fast=True):
	if Y.ndim == 1: 
		Y = Y[np.newaxis, :] # 1 x n
	(d, n) = Y.shape
	L = len(lambdas)

	if mask is None:
		mask = np.full((d, n), True)

	x = cp.Variable((d, 1))
	b = cp.Variable((d, n))

	lda = cp.Parameter(nonneg=True)
	x_broadcast = cp.kron(np.ones((1,n)), x)
	# for the second (L2) term, the mask shouldn't matter, since b_val is set to 0 in optimization
	obj = cp.sum_squares( cp.multiply(Y - x_broadcast - b, mask) ) + lda * cp.sum_squares(cp.multiply(b, mask)) 

	# construct constraints
	inequalities = []
	if ordering is not None:
		if ordering.ndim == 1:
			ordering = ordering[np.newaxis, :] # 1 x n

		assert(is_valid_ordering(ordering))
		
		vmax = int(np.max(ordering))
		for v in range(vmax+1): # find the closest smaller in train
			(xs_high, ys_high) = np.where( np.logical_and(ordering == v, mask) )
			size_high = len(xs_high)

			if size_high == 0: continue
			if mode_fast: # b ordering same as y ordering within a single course, single group
				for i in range(d):
					ys_high_block = ys_high[xs_high==i]
					y_block = Y[i, ys_high_block]
					if len(y_block) >= 2:
						idxs_order = np.argsort(y_block)
						for k in range(len(y_block)-1):
							inequalities.append(b[i, ys_high_block[idxs_order[k]]] <= b[i, ys_high_block[idxs_order[k+1]]])

			mask_low = np.logical_and(ordering < v, mask)
			if not np.any(mask_low): continue

			v_low = np.max(ordering[mask_low])
			(xs_low, ys_low) = np.where( np.logical_and(ordering == v_low, mask) )

			size_low = len(xs_low)

			if not mode_fast:
				for ih in range(size_high):
					for i in range(size_low):
						xh = xs_high[ih]
						yh = ys_high[ih]
						xl = xs_low[i]
						yl = ys_low[i]
						inequalities.append(b[xh, yh] >= b[xl, yl])
			else: # mode_fast
				for ih in range(d):
					for i in range(d):
						ys_high_block = ys_high[xs_high == ih]
						ys_low_block = ys_low[xs_low == i]

						if len(ys_high_block) > 0 and len(ys_low_block) > 0:
							y_high_block = Y[ih, ys_high_block]
							idx_high = np.argsort(y_high_block)[0] # min

							y_low_block = Y[i, ys_low_block]
							idx_low = np.argsort(y_low_block)[-1] # max

							inequalities.append(b[ih, ys_high_block[idx_high]] >= b[i, ys_low_block[idx_low]])

	# dummy constraints for the validation data to be 0
	(xs, ys) = np.where(np.logical_not(mask))
	for i in range(len(xs)):
		inequalities.append(b[xs[i], ys[i]] == 0)

	xs_sol = np.zeros((d, L))
	bs_sol = np.zeros((d, n, L))

	for il in range(L):
		l = lambdas[il]

		if l == np.inf:
			xs_sol[:, il] = np.sum(Y * mask, axis=1) / np.sum(mask, axis=1) # sample mean
			bs_sol[:, :, il] = np.zeros((d, n))
		else:
			lda.value = l

			if len(inequalities) == 0:
				prob = cp.Problem( cp.Minimize(obj))
			else:
				prob = cp.Problem( cp.Minimize(obj), inequalities)

			try:
				prob.solve(solver=cp.ECOS)
			except:
				print('Solving error (lambda=%.3f): %s' % (lambdas[il], sys.exc_info()[0]) )
				prob.solve(solver=cp.SCS)

			if l == 0: # break ties -- find the correct shift among all solutions
				b0 = b.value + x.value # broadcast operation

				x0 = cp.Variable((d, 1))
				x0_broadcast = cp.kron(np.ones((1,n)), x0)

				obj0 = cp.sum_squares( cp.multiply(b0 - x0_broadcast, mask) )
				inequalities0 = []

				# re-construct the inequalities again
				if ordering is not None:
					for v in range(vmax+1):
						(xs_high, ys_high) = np.where( np.logical_and(ordering == v, mask) )
						size_high = len(xs_high)

						# no need for constraints for a single course, single group, because it's already enforced in the first optimization
						mask_low = np.logical_and(ordering < v, mask)
						if size_high == 0 or not np.any(mask_low):
							continue

						v_low = np.max(ordering[mask_low])
						(xs_low, ys_low) = np.where( np.logical_and(ordering == v_low, mask) )

						size_low = len(xs_low)

						if not mode_fast:
							for ih in range(size_high):
								for i in range(size_low):
									xh = xs_high[ih]
									yh = ys_high[ih]
									xl = xs_low[i]
									yl = ys_low[i]
									if xh != xl: # address numerical infeasibility issue -- within course ordering constraints have already been resolved
										inequalities0.append(b0[xh, yh] - x0[xh] >= b0[xl, yl] - x0[xl])
						else: # mode_fast
							for ih in range(d):
								for i in range(d):
									if ih == i: continue # handled by previous optimization
									ys_high_block = ys_high[xs_high == ih]
									ys_low_block = ys_low[xs_low == i]

									if len(ys_high_block) > 0 and len(ys_low_block) > 0:
										y_high_block = b0[ih, ys_high_block]
										idx_high = np.argsort(y_high_block)[0] # min

										y_low_block = b0[i, ys_low_block]
										idx_low = np.argsort(y_low_block)[-1] # max

										inequalities0.append(b0[ih, ys_high_block[idx_high]] - x0[ih] >= b0[i, ys_low_block[idx_low]] - x0[i])

				prob0 = cp.Problem( cp.Minimize(obj0), inequalities0)

				try:
					prob0.solve(solver=cp.ECOS)
				except:
					print('Solving error (lambda=0).')
					prob0.solve(solver=cp.SCS)

				xs_sol[:, il] = x0.value.flatten()
				bs_sol[:, :, il] = b0 - x0.value

			else: # l > 0
				xs_sol[:, il] = x.value.flatten()
				bs_sol[:, :, il] = b.value

	return (xs_sol, bs_sol)

# Sample total ordering from partial ordering
# Returns:
# 	1x(nd) order (from small to large entries)
def sample_total_from_partial(d, n, ordering=None):
	if ordering is None:
		return np.random.permutation(d*n)
	else: # ordering not None
		assert(is_valid_ordering(ordering))
		order = ordering.flatten()
		order = order + 0.1 * np.random.uniform(size=d*n)
		idxs = np.argsort(order)
		return idxs

# From any values to adjacent-valued integers preserving the same rank, starting from 0:
def is_valid_ordering(ordering):
	vals = np.unique(ordering)
	if np.min(ordering) == 0:
		if np.all(np.diff(vals) == 1):
			return True
	return False
