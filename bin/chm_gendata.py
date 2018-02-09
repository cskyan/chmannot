#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: chm_gendata.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-03-01 22:15:59
###########################################################################
#

import os
import sys
import logging
import ast
from optparse import OptionParser

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD

import bionlp.spider.pubmed as pm
import bionlp.spider.metamap as mm
from bionlp import ftslct, ftdecomp
from bionlp.util import fs, io, sampling

import hoc

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
SPDR_MAP = {'hoc':hoc, 'pbmd':pm}
SC=';;'

opts, args = {}, []
cfgr = None
spdr = pm


def gen_data():
	if (opts.local):
		X, Y = spdr.get_data(None, from_file=True)
	else:
		pmid_list = spdr.get_pmids()
		articles = spdr.fetch_artcls(pmid_list)
		X, Y = spdr.get_data(articles, ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), fmt=opts.fmt, spfmt=opts.spfmt)
	hallmarks = Y.columns

	# Feature Selection
	# mt = sp.sparse.coo_matrix(X)
	# mask_mt = np.zeros(mt.shape)
	# mask_mt[mt.row, mt.col] = 1
	# stat = mask_mt.sum(axis=0)
	# cln_X = X.iloc[:,np.arange(stat.shape[0])[stat>ast.literal_eval(opts.thrshd) * (stat.max() - stat.min()) + stat.min()]]
	
	# Document Frequence
	# stat, _ = ftslct.freqs(X.values, Y.values)
	# Mutual information
	# stat, _ = ftslct.mutual_info(X.values, Y.values)
	# Information gain
	# stat, _ = ftslct.info_gain(X.values, Y.values)
	# GSS coefficient
	# stat, _ = ftslct.gss_coef(X.values, Y.values)
	# NGL coefficient
	# stat, _ = ftslct.ngl_coef(X.values, Y.values)
	# Odds ratio
	# stat, _ = ftslct.odds_ratio(X.values, Y.values)
	# Fisher criterion
	# stat, _ = ftslct.fisher_crtrn(X.values, Y.values)
	# GU metric
	# stat, _ = ftslct.gu_metric(X.values, Y.values)
	# Decision tree
	# stat, _ = ftslct.decision_tree(X.values, Y.values)
	# Combined feature
	stat, _ = ftslct.utopk(X.values, Y.values, ftslct.decision_tree, fn=100)
	io.write_npz(stat, os.path.join(spdr.DATA_PATH, 'ftw.npz'))
	
	# cln_X = X.iloc[:,np.arange(stat.shape[0])[stat>stat.min()]]
	cln_X = X.iloc[:,stat.argsort()[-500:][::-1]]
	print 'The size of data has been changed from %s to %s.' % (X.shape, cln_X.shape)
	
	if (opts.fmt == 'npz'):
		io.write_df(cln_X, os.path.join(spdr.DATA_PATH, 'cln_X.npz'), with_idx=True, sparse_fmt=opts.spfmt, compress=True)
	else:
		cln_X.to_csv(os.path.join(spdr.DATA_PATH, 'cln_X.csv'), encoding='utf8')
	del X, cln_X
	for i in range(Y.shape[1]):
		y = Y.iloc[:,i]
		if (opts.fmt == 'npz'):
			io.write_df(y, os.path.join(spdr.DATA_PATH, 'y_%s.npz' % i), with_col=False, with_idx=True)
		else:
			y.to_csv(os.path.join(spdr.DATA_PATH, 'y_%s.csv' % i), encoding='utf8')
		

def samp_data(sp_size = 0.3):
	pid = opts.pid
	if (pid != None):
		iter_size = 30
		X_iter, labels= spdr.get_feats_iter('y_%s.csv' % pid, iter_size)
		new_X, new_y = sampling.samp_df_iter(X_iter, iter_size, labels, sp_size)
		new_X.to_csv(os.path.join(spdr.DATA_PATH, 'samp_X_%i.csv' % pid), encoding='utf8')
		new_X.to_csv(os.path.join(spdr.DATA_PATH, 'samp_y_%s.csv' % pid), encoding='utf8')
	else:
		for i in range(10):
			iter_size = 30
			X_iter, labels= spdr.get_feats_iter('y_%s.csv' % i, iter_size)
			new_X, new_y = sampling.samp_df_iter(X_iter, iter_size, labels, sp_size)
			new_X.to_csv(os.path.join(spdr.DATA_PATH, 'samp_X_%i.csv' % i), encoding='utf8')
			new_X.to_csv(os.path.join(spdr.DATA_PATH, 'samp_y_%s.csv' % i), encoding='utf8')
			
			
def extend_mesh(ft_type='binary'):
	X, Y = spdr.get_data(None, ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), from_file=True, fmt=opts.fmt, spfmt=opts.spfmt)
	mesh_df = mm.mesh_countvec(X.index)
	mesh_df.columns = ['extmesh_' + x for x in mesh_df.columns]
	new_X = pd.concat([X, mesh_df], axis=1, join_axes=[X.index])
	print 'The size of data has been changed from %s to %s.' % (X.shape, new_X.shape)
	if (opts.fmt == 'npz'):
		io.write_df(new_X, os.path.join(spdr.DATA_PATH, 'extmesh_X.npz'), with_idx=True, sparse_fmt=opts.spfmt, compress=True)
	else:
		new_X.to_csv(os.path.join(spdr.DATA_PATH, 'extmesh_X.csv'), encoding='utf8')

	
def expand_data(ft_type='binary', db_name='mesh2016', db_type='LevelDB', store_path='store'):
	from rdflib import Graph
	from bionlp.util import ontology
	
	X, Y = spdr.get_data(None, ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), from_file=True, fmt=opts.fmt, spfmt=opts.spfmt)
	mesh_cols = filter(lambda x: x.startswith('mesh_') or x.startswith('extmesh_'), X.columns)
	mesh_X = X.loc[:,mesh_cols]
	exp_meshx = set([])
	ext_meshx_dict = {}
	g = Graph(store=db_type, identifier=db_name)
	g.open(store_path)
	for col in mesh_X.columns:
		mesh_lb = col.strip('extmesh_').strip('mesh_').replace('"', '\\"')
		# Get similar MeSH terms
		em_set = set(ontology.slct_sim_terms(g, mesh_lb, prdns=[('meshv',ontology.MESHV)], eqprds=ontology.MESH_EQPRDC_MAP))
		# Overall extended MeSH terms
		exp_meshx |= em_set
		# Extended MeSH terms per column
		ext_meshx_dict[col] = em_set
	g.close()
	exp_mesh_X = pd.DataFrame(np.zeros((mesh_X.shape[0], len(exp_meshx)), dtype='int8'), index=X.index, columns=['expmesh_%s' % w for w in exp_meshx])
	# Append the similar MeSH terms of each column to the final matrix
	for col, sim_mesh in ext_meshx_dict.iteritems():
		if (len(sim_mesh) == 0): continue
		sim_cols = ['expmesh_%s' % w for w in sim_mesh]
		if (ft_type == 'binary'):
			exp_mesh_X.loc[:,sim_cols] = np.logical_or(exp_mesh_X.loc[:,sim_cols], mesh_X.loc[:,col].reshape((-1,1))).astype('int')
		elif (ft_type == 'numeric'):
			exp_mesh_X.loc[:,sim_cols] += mesh_X.loc[:,col].reshape((-1,1))
		elif (ft_type == 'tfidf'):
			pass
	new_X = pd.concat([X, exp_mesh_X], axis=1, join_axes=[X.index])
	print 'The size of data has been changed from %s to %s.' % (X.shape, new_X.shape)
	if (opts.fmt == 'npz'):
		io.write_df(new_X, os.path.join(spdr.DATA_PATH, 'exp_X.npz'), with_idx=True, sparse_fmt=opts.spfmt, compress=True)
	else:
		new_X.to_csv(os.path.join(spdr.DATA_PATH, 'exp_X.csv'), encoding='utf8')
		
		
def decomp_data(method='LDA', n_components=100):
	X, Y = spdr.get_data(None, ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), from_file=True, fmt=opts.fmt, spfmt=opts.spfmt)
	method = method.upper()
	n_components = min(n_components, X.shape[1])
	if (method == 'LDA'):
		model = make_pipeline(LatentDirichletAllocation(n_topics=n_components, learning_method='online', learning_offset=50., max_iter=5, n_jobs=opts.np, random_state=0), Normalizer(copy=False))
	elif (method == 'NMF'):
		model = make_pipeline(NMF(n_components=n_components, random_state=0, alpha=.1, l1_ratio=.5), Normalizer(copy=False))
	elif (method == 'LSI'):
		model = make_pipeline(TruncatedSVD(n_components), Normalizer(copy=False))
	elif (method == 'TSNE'):
		model = make_pipeline(ftdecomp.DecompTransformer(n_components, ftdecomp.t_sne, initial_dims=15*n_components, perplexity=30.0))
	if (opts.prefix == 'all'):
		td_cols = X.columns
	else:
		# Only apply dimension reduction on specific columns
		td_cols = np.array(map(lambda x: True if any(x.startswith(prefix) for prefix in opts.prefix.split(SC)) else False, X.columns))
	td_X = X.loc[:,td_cols]
	new_td_X = model.fit_transform(td_X.as_matrix())
	if (opts.prefix == 'all'):
		columns = range(new_td_X.shape[1]) if not hasattr(model.steps[0][1], 'components_') else td_X.columns[model.steps[0][1].components_.argmax(axis=1)]
		new_X = pd.DataFrame(new_td_X, index=X.index, columns=['tp_%s' % x for x in columns])
	else:
		columns = range(new_td_X.shape[1]) if not hasattr(model.steps[0][1], 'components_') else td_X.columns[model.steps[0][1].components_.argmax(axis=1)]
		# Concatenate the components and the columns are not applied dimension reduction on
		new_X = pd.concat([pd.DataFrame(new_td_X, index=X.index, columns=['tp_%s' % x for x in columns]), X.loc[:,np.logical_not(td_cols)]], axis=1)
	if (opts.fmt == 'npz'):
		io.write_df(new_X, os.path.join(spdr.DATA_PATH, '%s%i_X.npz' % (method.lower(), n_components)), with_idx=True, sparse_fmt=opts.spfmt, compress=True)
	else:
		new_X.to_csv(os.path.join(spdr.DATA_PATH, '%s%i_X.csv' % (method.lower(), n_components)), encoding='utf8')
		
		
def add_d2v(n_components=100, win_size=8, min_t=5, mdl_fname='d2v.mdl'):
	from gensim.parsing.preprocessing import preprocess_string
	from gensim.models.doc2vec import TaggedDocument, Doc2Vec
	def read_files(fpaths, code='ascii'):
		for fpath in fpaths:
			try:
				yield TaggedDocument(words=preprocess_string('\n'.join(fs.read_file(fpath, code))), tags=[os.path.splitext(os.path.basename(fpath))[0]])
			except Exception as e:
				continue
	def read_prcsed_files(fpaths, code='ascii'):
		for fpath in fpaths:
			try:
				words = []
				for line in fs.read_file(fpath, code):
					if (line == '~~~'):
						continue
					if (line == '.	.	.' or line == '~~~	~~~' or line == ',	,	,'):
						continue
					items = line.split()
					if (len(items) < 3): # Skip the unrecognized words
						continue
					words.append(items[2].lower())
				yield TaggedDocument(words=words, tags=[os.path.splitext(os.path.basename(fpath))[0]])
			except Exception as e:
				continue
	mdl_fpath = os.path.join(spdr.DATA_PATH, mdl_fname)
	if (os.path.exists(mdl_fpath)):
		model = Doc2Vec.load(mdl_fpath)
	else:
		# model = Doc2Vec(read_files(fs.listf(spdr.ABS_PATH, full_path=True)), size=n_components, window=8, min_count=5, workers=opts.np)
		model = Doc2Vec(read_prcsed_files(fs.listf(os.path.join(spdr.DATA_PATH, 'lem'), full_path=True)), size=n_components, window=8, min_count=5, workers=opts.np)
		model.save(os.path.join(spdr.DATA_PATH, mdl_fname))
		
	X, Y = spdr.get_data(None, ft_type=opts.type, max_df=ast.literal_eval(opts.maxdf), min_df=ast.literal_eval(opts.mindf), from_file=True, fmt=opts.fmt, spfmt=opts.spfmt)
	# Map the index of original matrix to that of the paragraph vectors
	d2v_idx = [model.docvecs.index_to_doctag(i).rstrip('.lem') for i in range(model.docvecs.count)]
	mms = MinMaxScaler()
	d2v_X = pd.DataFrame(mms.fit_transform(model.docvecs[range(model.docvecs.count)]), index=d2v_idx, columns=['d2v_%i' % i for i in range(model.docvecs[0].shape[0])])
	# d2v_X = pd.DataFrame(model.docvecs[range(model.docvecs.count)], index=d2v_idx, columns=['d2v_%i' % i for i in range(model.docvecs[0].shape[0])])
	new_X = pd.concat([X, d2v_X], axis=1, join_axes=[X.index])
	print 'The size of data has been changed from %s to %s.' % (X.shape, new_X.shape)
	if (opts.fmt == 'npz'):
		io.write_df(d2v_X, os.path.join(spdr.DATA_PATH, 'd2v_X.npz'), with_idx=True, sparse_fmt=opts.spfmt, compress=True)
		io.write_df(new_X, os.path.join(spdr.DATA_PATH, 'cmb_d2v_X.npz'), with_idx=True, sparse_fmt=opts.spfmt, compress=True)
	else:
		d2v_X.to_csv(os.path.join(spdr.DATA_PATH, 'd2v_X.csv'), encoding='utf8')
		new_X.to_csv(os.path.join(spdr.DATA_PATH, 'cmb_d2v_X.csv'), encoding='utf8')

	
def main():
	if (opts.method is None):
		return
	elif (opts.method == 'gen'):
		gen_data()
	elif (opts.method == 'samp'):
		samp_data()
	elif (opts.method == 'extend'):
		extend_mesh()
	elif (opts.method == 'expand'):
		expand_data(store_path=os.path.join(spdr.DATA_PATH, 'store'))
	elif (opts.method == 'decomp'):
		decomp_data(method=opts.decomp.upper(), n_components=opts.cmpn)
	elif (opts.method == 'd2v'):
		add_d2v(n_components=opts.cmpn)
	

if __name__ == '__main__':
	# Logging setting
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

	# Parse commandline arguments
	op = OptionParser()
	op.add_option('-p', '--pid', action='store', type='int', dest='pid', help='indicate the process ID')
	op.add_option('-n', '--np', default=-1, action='store', type='int', dest='np', help='indicate the number of processes used for training')
	op.add_option('-f', '--fmt', default='npz', help='data stored format: csv or npz [default: %default]')
	op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csr or csc [default: %default]')
	op.add_option('-l', '--local', default=False, action='store_true', dest='local', help='read data from the preprocessed data matrix file')
	op.add_option('-t', '--type', default='binary', help='feature type: binary, numeric, tfidf [default: %default]')
	op.add_option('-a', '--mindf', default='1', type='str', dest='mindf', help='lower document frequency threshold for term ignorance')
	op.add_option('-b', '--maxdf', default='1.0', type='str', dest='maxdf', help='upper document frequency threshold for term ignorance')
	op.add_option('-r', '--thrshd', default='0.05', type='str', dest='thrshd', help='feature frequency threshold for filtering')
	op.add_option('-d', '--decomp', default='LDA', help='decomposition method to use: LDA, NMF, LSI or TSNE [default: %default]')
	op.add_option('-c', '--cmpn', default=100, type='int', dest='cmpn', help='number of components that used in clustering model')
	op.add_option('-j', '--prefix', default='all', type='str', dest='prefix', help='prefixes of the column names that the decomposition method acts on, for example, \'-j lem;;nn;;ner\' means columns that starts with \'lem_\', \'nn_\', or \'ner_\'')
	op.add_option('-i', '--input', default='hoc', help='input source: hoc or pbmd [default: %default]')
	op.add_option('-m', '--method', help='main method to run')

	(opts, args) = op.parse_args()
	if len(args) > 0:
		op.print_help()
		op.error('Please input options instead of arguments.')
		sys.exit(1)

	spdr = SPDR_MAP[opts.input]
	# Parse config file
	if (os.path.exists(CONFIG_FILE)):
		cfgr = io.cfg_reader(CONFIG_FILE)
		spdr_cfg = cfgr('bionlp.spider.%s' % opts.input, 'init')
		if (len(spdr_cfg) > 0 and spdr_cfg['DATA_PATH'] is not None and os.path.exists(spdr_cfg['DATA_PATH'])):
			spdr.DATA_PATH = spdr_cfg['DATA_PATH']

	main()