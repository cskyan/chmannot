#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: chm_helper.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-04-10 14:55:16
###########################################################################
#

import os
import sys
import ast
import glob
import logging
import itertools
from optparse import OptionParser

import numpy as np
import scipy as sp
import pandas as pd

from bionlp.util import fs, io, plot

import hoc

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
HMLB = ['PS', 'GS', 'CD', 'RI', 'A', 'IM', 'GI', 'TPI', 'CE', 'ID']
SC=';;'

opts, args = {}, []
cfgr = None


def post_process(spdr):
	if (opts.fmt == 'csv'):
		X, Y = spdr.get_feats(['y_%s'%i for i in range(10)])
	elif (opts.fmt == 'npz'):
		X, Y = spdr.get_feats_npz(['y_%s'%i for i in range(10)])
	else:
		print 'Unsupported format: %s' % opts.fmt
		sys.exit(1)

		
def npzs2yaml(dir_path='.', mdl_t='Classifier'):
	pw = io.param_writer(os.path.join(dir_path, 'mdlcfg'))
	for file in fs.listf(dir_path):
		if file.endswith(".npz"):
			fpath = os.path.join(dir_path, file)
			params = io.read_npz(fpath)['best_params'].tolist()
			for k in params.keys():
				if (type(params[k]) == np.ndarray):
					params[k] == params[k].tolist()
				if (isinstance(params[k], np.generic)):
					params[k] = np.asscalar(params[k])
			pw(mdl_t, file, params)
	pw(None, None, None, True)
	

def merge_param_1d(fpath1, fpath2):
	''' Merge the featfilt__k parameter in two tuning npz file'''
	npzf1, npzf2 = io.read_npz(fpath1), io.read_npz(fpath2)
	data = dict(npzf1.items()), dict(npzf2.items())
	dim_vals1, dim_vals2 = data[0]['dim_vals'].tolist()['featfilt__k'], data[1]['dim_vals'].tolist()['featfilt__k']
	vals = dim_vals1.keys() + dim_vals2.keys()
	dim = [(0,i) for i in dim_vals1.values()] + [(1,i) for i in dim_vals2.values()]
	sorted_vals, sorted_idx = np.unique(vals, return_index=True)
	dim_vals = dict(featfilt__k=dict([(k, i) for i, k in enumerate(sorted_vals)]))
	score_avg_cube = [data[dim[idx][0]]['score_avg_cube'][dim[idx][1]] for idx in sorted_idx]
	score_std_cube = [data[dim[idx][0]]['score_std_cube'][dim[idx][1]] for idx in sorted_idx]
	max_idx = np.argmax(score_avg_cube)
	best_params = dict(featfilt__k=sorted_vals[max_idx])
	best_score = score_avg_cube[max_idx]
	io.write_npz(dict(zip(['best_params', 'best_score', 'score_avg_cube', 'score_std_cube', 'dim_names', 'dim_vals'], (best_params, best_score, score_avg_cube, score_std_cube, data[0]['dim_names'], dim_vals))), os.path.basename(fpath1))
	
	
def npzs2ftkcht(dir_path='.'):
	data, labels = [[] for i in range(2)]
	for file in fs.listf(dir_path):
		subdata_list = []
		if file.endswith(".npz"):
			npzf = io.read_npz(os.path.join(dir_path, file))
			dim_vals = npzf['dim_vals'].tolist()['featfilt__k']
			vals, val_ids = np.array(dim_vals.keys()), np.array(dim_vals.values())
			sorted_idx = vals.argsort()
			data.append((vals[sorted_idx], npzf['score_avg_cube'][val_ids[sorted_idx]]))
		labels.append(os.path.splitext(file)[0])
	plot.plot_ftnum(data, labels, marker=True)
	
	
def avgfeatw(dir_path='.'):
	df_list = []
	for file in fs.listf(dir_path):
		if file.endswith(".npz"):
			df_list.append(io.read_df(os.path.join(dir_path, file), with_idx=True))
	feat_w_mt = pd.concat([df.loc[:,'Importance Mean'] for df in df_list], axis=1, join_axes=[df_list[0].index]).astype('float').values
	feat_w_avg = feat_w_mt.mean(axis=1)
	feat_w_std = feat_w_mt.std(axis=1)
	sorted_idx = np.argsort(feat_w_avg, axis=-1)[::-1]
	sorted_feat_w = np.column_stack((df_list[0].loc[:,'Feature Name'].values[sorted_idx], feat_w_avg[sorted_idx], feat_w_std[sorted_idx]))
	feat_w_df = pd.DataFrame(sorted_feat_w, index=df_list[0].index.values[sorted_idx], columns=['Feature Name', 'Importance Mean', 'Importance Std'])
	feat_w_df.to_excel(os.path.join(dir_path, 'featw.xlsx'))
	io.write_df(feat_w_df, os.path.join(dir_path, 'featw'), with_idx=True)

	
def pred2cor(dir_path, file_ptn, mdls, pids=range(10), crsval=10):
	import scipy.stats as stats
	from chm_annot import pred_ovl
	import bionlp.util.math as imath
	for pid in pids:
		crsval_povl, crsval_spearman = [[] for i in range(2)]
		for crs_t in xrange(crsval):
			preds, true_lb = [], None
			for mdl in mdls:
				mdl = mdl.replace(' ', '_').lower()
				file = file_ptn.replace('#CRST#', str(crs_t)).replace('#MDL#', mdl).replace('#PID#', str(pid))
				npz_file = io.read_npz(os.path.join(dir_path, file))
				preds.append(npz_file['pred_lb'])
				true_lb = npz_file['true_lb']
			preds_mt = np.column_stack([x.ravel() for x in preds])
			preds.append(true_lb)
			tpreds_mt = np.column_stack([x.ravel() for x in preds])
			crsval_povl.append(pred_ovl(preds_mt, true_lb.ravel()))
			crsval_spearman.append(stats.spearmanr(tpreds_mt))
		povl_avg = np.array(crsval_povl).mean(axis=0).round()
		spmnr_avg = np.array([crsp[0] for crsp in crsval_spearman]).mean(axis=0)
		spmnr_pval = np.array([crsp[1] for crsp in crsval_spearman]).mean(axis=0)
		povl_idx = list(imath.subset(mdls, min_crdnl=1))
		povl_avg_df = pd.DataFrame(povl_avg, index=povl_idx, columns=['pred_ovl', 'tpred_ovl'])
		spmnr_avg_df = pd.DataFrame(spmnr_avg, index=mdls+['Annotations'], columns=mdls+['Annotations'])
		spmnr_pval_df = pd.DataFrame(spmnr_pval, index=mdls+['Annotations'], columns=mdls+['Annotations'])
		povl_avg_df.to_excel(os.path.join(dir_path, 'cpovl_avg_%s.xlsx' % pid))
		spmnr_avg_df.to_excel(os.path.join(dir_path, 'spmnr_avg_%s.xlsx' % pid))
		spmnr_pval_df.to_excel(os.path.join(dir_path, 'spmnr_pval_%s.xlsx' % pid))
		io.write_df(povl_avg_df, os.path.join(dir_path, 'povl_avg_%s.npz' % pid), with_idx=True)
		io.write_df(spmnr_avg_df, os.path.join(dir_path, 'spmnr_avg_%s.npz' % pid), with_idx=True)
		io.write_df(spmnr_pval_df, os.path.join(dir_path, 'spmnr_val_%s.npz' % pid), with_idx=True)
		
		
def pred2uniq(dir_path, file_ptn, mdls, pids=range(10), crsval=10):
	import scipy.stats as stats
	from hm_clf import pred_ovl
	import bionlp.util.math as imath
	# uniqp_dict = dict.fromkeys(mdls, dict.fromkeys(HMLB, []))
	uniqp_dict = dict((mdl, dict((k, []) for k in HMLB)) for mdl in mdls)
	for pid in pids:
		crsval_povl, crsval_spearman = [[] for i in range(2)]
		for crs_t in xrange(crsval):
			preds, true_lb = [], None
			for mdl in mdls:
				mdl = mdl.replace(' ', '_').lower()
				file = file_ptn.replace('#CRST#', str(crs_t)).replace('#MDL#', mdl).replace('#PID#', str(pid))
				npz_file = io.read_npz(os.path.join(dir_path, file))
				preds.append(npz_file['pred_lb'])
				true_lb = npz_file['true_lb']
			preds.append(true_lb)
			tpreds_mt = np.column_stack([x.ravel() for x in preds])
			test_idx = io.read_df(os.path.join(dir_path, 'test_idx_crsval_%s_%s.npz' % (crs_t, pid)), with_idx=True).index.tolist()
			uniq_true_list = []
			for i in xrange(tpreds_mt.shape[1] - 1):
				rmd_idx = range(tpreds_mt.shape[1] - 1)
				del rmd_idx[i]
				condition = np.logical_and(np.logical_not(np.any(tpreds_mt[:,rmd_idx], axis=1)), np.all(tpreds_mt[:,[i,-1]], axis=1))
				uniq_true = np.arange(tpreds_mt.shape[0])[condition]
				for j in uniq_true:
					uniqp_dict[mdls[i]][HMLB[j % true_lb.shape[1]]].append(test_idx[j / true_lb.shape[1]])
				uniq_true_list.append(', '.join(['%s | %s' % (test_idx[x/true_lb.shape[1]], x%true_lb.shape[1]) for x in uniq_true]))
			uniq_true_str = '\n'.join(['%s: %s' % (mdls[i], uniq_true_list[i]) for i in range(len(uniq_true_list))])
			fs.write_file(uniq_true_str, os.path.join(dir_path, 'uniqtrue_crsval_%s_%s.txt' % (crs_t, pid)))
	for mdl, idmap in uniqp_dict.iteritems():
		uniqp_df = pd.DataFrame([(hm, ', '.join(idmap[hm])) for hm in HMLB], columns=['Hallmark', 'PMIDS'])
		uniqp_df.to_excel(os.path.join(dir_path, 'uniqtrue_%s.xlsx' % mdl), index=False)


def nt2db():
	import bionlp.util.ontology as ontology
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	ontology.files2db(opts.loc, fmt='nt', saved_path=os.path.splitext(opts.loc)[0] if opts.output is None else opts.output, **kwargs)
	
	
def nt2xml(dir_path='.'):
	import rdflib
	nt_file = 'mesh2016_samp.nt'
	rdf_name = os.path.splitext(nt_file)[0]
	print 'Creating RDF XML %s.rdf in %s...' % (rdf_name, dir_path)
	graph = rdflib.Graph(identifier=rdf_name)
	graph.parse(os.path.join(dir_path, nt_file), format='nt')
	with open(os.path.join(dir_path, '%s.rdf'%rdf_name), 'w') as fd:
		graph.serialize(fd, format="xml")
	graph.close()
	
	
def axstat(spdr):
	data_path = spdr.DATA_PATH
	mt_file = 'X.npz'
	import bionlp.util.io as io
	mt = sp.sparse.coo_matrix(io.read_df(os.path.join(data_path, mt_file), sparse_fmt='csr').as_matrix())
	mask_mt = np.zeros(mt.shape)
	mask_mt[mt.row, mt.col] = 1
	axstat0 = mask_mt.sum(axis=0)
	plot.plot_hist(axstat0, '# abstracts', '# features', log=True, title='Number of features (y-axis) that have m abstracts (x-axis)', fname='hist_abs_ft')
#	hist0 = np.histogram(axstat0, bins=max(10, min(20, int(axstat0.max()))))
#	plot.plot_scat(np.column_stack((hist0[1][:-1], hist0[0])), '# abstracts', '# features', title='Number of features (y-axis) that have m abstracts (x-axis)', fname='scat_abs_ft')
	axstat1 = mask_mt.sum(axis=1)
	plot.plot_hist(axstat1, '# features', '# abstracts', log=False, title='Number of abstracts (y-axis) that have n features (x-axis)', fname='hist_ft_abs')
	
	
def axstat_cmp(spdr):
	data_path = spdr.DATA_PATH
	mt1_file, mt2_file = 'X1.npz', 'X2.npz'
	import bionlp.util.io as io
	mt1 = sp.sparse.coo_matrix(io.read_df(os.path.join(data_path, mt1_file), sparse_fmt='csr').as_matrix())
	mt2 = sp.sparse.coo_matrix(io.read_df(os.path.join(data_path, mt2_file), sparse_fmt='csr').as_matrix())
	mask1_mt = np.zeros(mt1.shape)
	mask2_mt = np.zeros(mt2.shape)
	mask1_mt[mt1.row, mt1.col] = 1
	mask2_mt[mt2.row, mt2.col] = 1
	axstat1 = mask1_mt.sum(axis=0)
	axstat2 = mask2_mt.sum(axis=0)
	plot.plot_2hist(axstat1, axstat2, '# abstracts', '# features through \nnormalized cumulative log', normed=True, cumulative=True, log=True, title='Number of features (y-axis) that have m abstracts (x-axis)', fname='hist_abs_ft')
	
	
def ftstat(spdr):
	ft_order = ['lem', 'nn', 'ner', 'parse', 'vc', 'mesh', 'chem']
	ft_name = {'lem':'LBoW', 'nn':'N-Bigrams', 'ner':'NE', 'parse':'GR', 'vc':'VC', 'mesh':'MeSH', 'chem':'Chem'}
#	ft_order = ['lem', 'nn', 'ner', 'parse', 'vc', 'mesh', 'chem', 'extmesh', 'expmesh']
#	ft_name = {'lem':'LBoW', 'nn':'N-Bigrams', 'ner':'NE', 'parse':'GR', 'vc':'VC', 'mesh':'MeSH', 'chem':'Chem', 'extmesh':'ExtMeSH', 'expmesh':'ExpMeSH'}
	data_path = spdr.DATA_PATH
	mt_file = 'X.npz', 'Y.npz'
	import bionlp.util.io as io
	X, Y = io.read_df(os.path.join(data_path, mt_file[0]), sparse_fmt='csr'), io.read_df(os.path.join(data_path, mt_file[1]), sparse_fmt='csr')
	ft_idx = {}
	for i, col in enumerate(X.columns):
		for ft in ft_order:
			if (col.startswith(ft+'_')):
				ft_idx.setdefault(ft, []).append(i)
				break
	ftor_list = []
	for i in xrange(Y.shape[1]):
#		mt_lb = sp.sparse.csr_matrix(X.iloc[np.arange(Y.shape[0])[Y.iloc[:,i].values == 1],:].as_matrix())
		agg_X = X.iloc[np.arange(Y.shape[0])[Y.iloc[:,i].values == 1],:].values.sum(axis=0)
		ft_sum = np.zeros((1,X.shape[1]))
#		ft_sum[0,mt_lb.indices] = 1
		ft_sum[0,agg_X > 0] = 1
		ftor_list.append(ft_sum)
	ftor_mt = np.concatenate(ftor_list, axis=0)
	ft_set_list = []
	for ft in ft_order:
		ft_set_list.append(ftor_mt[:,ft_idx[ft]].sum(axis=1))
	ft_stat_mt = np.column_stack(ft_set_list)
	ft_stat_pd = pd.DataFrame(ft_stat_mt, index=HMLB, columns=[ft_name[fset] for fset in ft_order])
	ft_stat_pd.to_excel('ft_stat.xlsx')
	
	
def param_analysis():
	kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
	param_name = kwargs['param']
	param_tuning = io.read_npz(opts.loc)
	dim_names, dim_vals, best_params, score_avg, score_std = param_tuning['dim_names'].tolist(), param_tuning['dim_vals'].tolist(), param_tuning['best_params'].tolist(), param_tuning['score_avg_cube'], param_tuning['score_std_cube']
	param_dim = dim_names[target_param]
	val, avg, std = analyze_param(param_name, score_avg, score_std, dim_names, dim_vals, best_params)
	plot.plot_param(val, avg, std, xlabel='Number of Decision Trees', ylabel='Micro F1 Score')
	
	
def matshow(spdr):
	from matplotlib import pyplot as plt
	data_path = spdr.DATA_PATH
	mt_file = 'X.npz'
	X=io.read_df(os.path.join(data_path, mt_file), with_idx=True, sparse_fmt='csr')
	plt.matshow(X.values, cmap=plt.cm.Blues)
	plt.title('Standard Dataset')
	plt.savefig('X_matshow')
	
	
def npz2xls(spdr):
	data_path = spdr.DATA_PATH
	npz_file = 'hm_stat.npz'
	fpath = os.path.join(data_path, npz_file)
	df = io.read_df(fpath)
	df.to_excel(os.path.splitext(fpath)[0] + '.xlsx')
	
	
def filtx(spdr):
	data_path = spdr.DATA_PATH
	mt_file = 'X.npz', 'Y.npz'
	import bionlp.util.io as io
	X, Y = io.read_df(os.path.join(data_path, mt_file[0]), with_idx=True, sparse_fmt='csr'), io.read_df(os.path.join(data_path, mt_file[1]), with_idx=True, sparse_fmt='csr')
	Xs = spdr.ft_filter(X, Y)
	for i, x_df in enumerate(Xs):
		io.write_df(x_df, os.path.join(data_path, 'X_%i.npz' % i), with_idx=True, sparse_fmt='csr', compress=True)
		

def unionx(spdr):
	data_path = spdr.DATA_PATH
	mt_file = 'X.npz', 'Y.npz'
	ft_order = ['lem', 'nn', 'ner', 'parse', 'vc', 'mesh', 'chem', 'extmesh']
	import bionlp.util.io as io
	X, Y = io.read_df(os.path.join(data_path, mt_file[0]), with_idx=True, sparse_fmt='csr'), io.read_df(os.path.join(data_path, mt_file[1]), with_idx=True, sparse_fmt='csr')
	ft_idx = {}
	for i, col in enumerate(X.columns):
		for ft in ft_order:
			if (col.startswith(ft+'_')):
				ft_idx.setdefault(ft, []).append(i)
				break
	union_ft_idx = {}
	for j in range(Y.shape[1]):
		X_j = io.read_df(os.path.join(data_path, 'X_%i.npz'%j), with_idx=True, sparse_fmt='csr')
		for i, col in enumerate(X_j.columns):
			for ft in ft_order:
				if (col.startswith(ft+'_')):
					union_ft_idx.setdefault(ft, set([])).add(col)
					break
	new_ft = []
	for ft in ft_order:
		new_ft.extend(list(union_ft_idx[ft]))
	union_X = X.loc[:,new_ft]
	io.write_df(union_X, os.path.join(data_path, 'union_X.npz'), with_idx=True, sparse_fmt='csr', compress=True)
		
		
def leave1out(spdr, mltl=True):
	data_path = spdr.DATA_PATH
	mt_file = 'X.npz', 'Y.npz'
	ft_order = ['lem', 'nn', 'ner', 'parse', 'vc', 'mesh', 'chem']
	import bionlp.util.io as io
	X, Y = io.read_df(os.path.join(data_path, mt_file[0]), with_idx=True, sparse_fmt='csr'), io.read_df(os.path.join(data_path, mt_file[1]), with_idx=True, sparse_fmt='csr')
	ft_dict = {}
	for col in X.columns:
		for ft in ft_order:
			if (col.startswith(ft+'_')):
				ft_dict.setdefault(ft, []).append(col)
				break
	if (mltl):
		for ft in ft_order:
			new_X = X.drop(ft_dict[ft], axis=1)
			io.write_df(new_X, os.path.join(data_path, 'l1o_%s_X.npz'%ft), sparse_fmt='csr', compress=True)
	else:
		for i in range(Y.shape[1]):
			X_i = io.read_df(os.path.join(data_path, 'X_%i.npz'%i), sparse_fmt='csr')
			for ft in ft_order:
				new_X = X_i.drop(ft_dict[ft], axis=1)
				io.write_df(new_X, os.path.join(data_path, 'l1o_%s_X_%i.npz'%(ft,i)), sparse_fmt='csr', compress=True)
				
				
def ftcor(dir_path='.'):
	axes_pair = [('DF', 'UDT'), ('DF', 'UNGL'), ('DF', 'UGSS')]
	axes = list(set(itertools.chain(*axes_pair)))
	axes_npzf = [io.read_npz(os.path.join(dir_path, ax+'.npz')) for ax in axes]
	for xlb, ylb in axes_pair:
		xnpzf, ynpzf = axes_npzf[axes.index(xlb)], axes_npzf[axes.index(ylb)]
		plot.plot_scat(np.column_stack((xnpzf['data'], ynpzf['data'])), xlb, ylb, scale=('symlog', None), title='Correlation between %s and %s values of features'%(xlb, ylb), fname='%s_%s_ftw_cor'%(xlb.lower(), ylb.lower()))
	

def annotfig(fpath):
	import matplotlib.pyplot as plt
	from mpldatacursor import datacursor
	plt.switch_backend('TkAgg')
	if (opts.ele != ''):
		annot_str, ref_str = opts.ele.split('#')
		annotx_str, annoty_str = annot_str.split(';')
		refx_str, refy_str = ref_str.split(';')
	annotation = {point:'' for point in zip([float(x) for x in annotx_str.split(',') if x], [float(y) for y in annoty_str.split(',') if y])}
	ref_lines = {'x':[float(x) for x in refx_str.split(',') if x], 'y':[float(y) for y in refy_str.split(',') if y]}
	plot_cfg = ast.literal_eval(opts.cfg) if (opts.cfg != '') else {}
	plot.plot_files(fpath, ref_lines=ref_lines, plot_cfg=plot_cfg, annotator=None, annotation=annotation)
	
	
def zoominfig(fpath):
	pass


def main():
	if (opts.method is None):
		return
	elif (opts.method == 'post'):
		post_process(hoc)
	elif (opts.method == 'n2y'):
		npzs2yaml(opts.loc)
	elif (opts.method == 'n2f'):
		npzs2ftkcht(opts.loc)
	elif (opts.method == 'avgf'):
		avgfeatw(opts.loc)
	elif (opts.method == 'p2c'):
#		pred2cor(opts.loc, 'pred_crsval_#CRST#_#MDL#_#PID#.npz', ['Random forest', 'LinearSVC with L1 penalty', 'LinearSVC with L2 penalty [Ft Filt] & Perceptron [CLF]', 'Extra Trees [Ft Filt] & LinearSVC with L1 penalty [CLF]'])
		pred2cor(opts.loc, 'pred_crsval_#CRST#_#MDL#_#PID#.npz', ['UDT-RF', 'DF-RbfSVM', 'MEM', 'MNB'], pids=['all'])
	elif (opts.method == 'p2u'):
		pred2uniq(opts.loc, 'pred_crsval_#CRST#_#MDL#_#PID#.npz', ['UDT-RF', 'DF-RbfSVM', 'MEM', 'MNB'], pids=['all'])
	elif (opts.method == 'n2d'):
		nt2db()
	elif (opts.method == 'n2x'):
		nt2xml(opts.loc)
	elif (opts.method == 'axstat'):
		axstat(hoc)
	elif (opts.method == 'axstat_cmp'):
		axstat_cmp(hoc)
	elif (opts.method == 'ftstat'):
		ftstat(hoc)
	elif (opts.method == 'matshow'):
		matshow(hoc)
	elif (opts.method == 'npz2xls'):
		npz2xls(hoc)
	elif (opts.method == 'filtx'):
		filtx(hoc)
	elif (opts.method == 'unionx'):
		unionx(hoc)
	elif (opts.method == 'l1o'):
		leave1out(hoc, True)
	elif (opts.method == 'ftcor'):
		ftcor(opts.loc)
	elif (opts.method == 'annotfig'):
		annotfig(opts.loc)
	elif (opts.method == 'mp1d'):
		merge_param_1d(*opts.loc.split(';;'))
	

if __name__ == '__main__':
	# Logging setting
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

	# Parse commandline arguments
	op = OptionParser()
	op.add_option('-f', '--fmt', default='npz', help='data stored format: csv or npz [default: %default]')
	op.add_option('-l', '--loc', default='.', help='the files in which location to be process')
	op.add_option('-o', '--output', default='.', help='the path to store the data')
	op.add_option('-e', '--ele', default='', help='plot elements string used in the function, format: annot_x0,...,annot_xn;annot_y0,...,annot_yn#ref_x0,...,ref_xn;ref_y0,...,ref_yn')
	op.add_option('-c', '--cfg', default='', help='config string used in the plot function, format: {\'param_name1\':param_value1[, \'param_name1\':param_value1]}')
	op.add_option('-m', '--method', help='main method to run')

	(opts, args) = op.parse_args()
	if len(args) > 0:
		op.print_help()
		op.error('Please input options instead of arguments.')
		sys.exit(1)
		
	# Parse config file
	if (os.path.exists(CONFIG_FILE)):
		cfgr = io.cfg_reader(CONFIG_FILE)
		hoc_cfg = cfgr('bionlp.spider.hoc', 'init')
		if (len(hoc_cfg) > 0 and hoc_cfg['DATA_PATH'] is not None and os.path.exists(hoc_cfg['DATA_PATH'])):
			hoc.DATA_PATH = hoc_cfg['DATA_PATH']

	main()