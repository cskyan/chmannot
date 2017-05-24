#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: chm_annot.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-03-16 15:56:16
###########################################################################
#

import os
import sys
import logging
from sets import Set
from optparse import OptionParser

import numpy as np
import scipy as sp
import scipy.stats as stats
import pandas as pd

from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, SelectFpr, SelectFromModel, chi2, f_classif
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier, LassoCV, LassoLarsCV, LassoLarsIC, RandomizedLasso
from sklearn.svm import SVC, LinearSVC
#from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from bionlp import ftslct, txtclf
from bionlp.util import io, func
import bionlp.spider.pubmed as pm

import hoc


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
SPDR_MAP = {'hoc':hoc, 'pbmd':pm}
FILT_NAMES, CLF_NAMES, PL_NAMES = [[] for i in range(3)]
PL_SET = Set([])
SC=';;'

opts, args = {}, []
cfgr = None
spdr = pm


def load_data(mltl=False, pid=0, spfmt='csr'):
	print 'Loading data...'
	try:
		if (mltl):
			# From combined data file
			X, Y = spdr.get_data(None, from_file=True, fmt=opts.fmt, spfmt=opts.spfmt)
			y = Y.as_matrix()
		else:
			# From splited data file
#			Xs, Ys = spdr.get_mltl([pid])
			Xs, Ys = spdr.get_mltl_npz(lbs=[pid], spfmt=spfmt)
			X = Xs[0]
			y = Ys[0].as_matrix().reshape(Ys[0].shape[0],)
	except Exception as e:
		print e
		print 'Can not find the data files!'
		exit(1)
	return X, y

def build_model(mdl_func, mdl_t, mdl_name, tuned=False, pr=None, mltl=False, **kwargs):
	if (tuned and bool(pr)==False):
		print 'Have not provided parameter writer!'
		return None
	if (mltl):
		return OneVsRestClassifier(mdl_func(**func.update_dict(pr(mdl_t, mdl_name) if tuned else {}, kwargs)), n_jobs=opts.np)
	else:
		return mdl_func(**func.update_dict(pr(mdl_t, mdl_name) if tuned else {}, kwargs))


# Feature Filtering Models
def gen_featfilt(tuned=False, glb_filtnames=[]):
	tuned = tuned or opts.best
	common_cfg = cfgr('chm_annot', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	filt_names = []
	for filt_name, filter in [
#		('Var Cut', VarianceThreshold()),
#		('Chi2 Pval on FPR', SelectFpr(chi2, alpha=0.05)),
#		('ANOVA-F Pval on FPR', SelectFpr(f_classif, alpha=0.05)),
#		('Chi2 Top K Perc', SelectPercentile(chi2, percentile=30)),
#		('ANOVA-F Top K Perc', SelectPercentile(f_classif, percentile=30)),
#		('Chi2 Top K', SelectKBest(chi2, k=1000)),
#		('ANOVA-F Top K', SelectKBest(f_classif, k=1000)),
#		('LinearSVC', LinearSVC(loss='squared_hinge', dual=False, **pr('Classifier', 'LinearSVC') if tuned else {})),
#		('Logistic Regression', SelectFromModel(LogisticRegression(dual=False, **pr('Feature Selection', 'Logistic Regression') if tuned else {}))),
#		('Lasso', SelectFromModel(LassoCV(cv=6), threshold=0.16)),
#		('Lasso-LARS', SelectFromModel(LassoLarsCV(cv=6))),
#		('Lasso-LARS-IC', SelectFromModel(LassoLarsIC(criterion='aic'), threshold=0.16)),
#		('Randomized Lasso', SelectFromModel(RandomizedLasso(random_state=0))),
#		('Extra Trees Regressor', SelectFromModel(ExtraTreesRegressor(100))),
		# ('U102-GSS502', ftslct.MSelectKBest(ftslct.gen_ftslct_func(ftslct.utopk, filtfunc=ftslct.gss_coef, fn=100), k=500)),
		# ('GSS502', ftslct.MSelectKBest(ftslct.gss_coef, k=500)),
#		('Combined Model', FeatureUnion([('Var Cut', VarianceThreshold()), ('Chi2 Top K', SelectKBest(chi2, k=1000))])),
		('No Feature Filtering', None)
	]:
		yield filt_name, filter
		filt_names.append(filt_name)
	if (len(glb_filtnames) < len(filt_names)):
		del glb_filtnames[:]
		glb_filtnames.extend(filt_names)


# Classification Models
def gen_clfs(tuned=False, glb_clfnames=[]):
	tuned = tuned or opts.best
	common_cfg = cfgr('chm_annot', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	clf_names = []
	for clf_name, clf in [
#		('RidgeClassifier', RidgeClassifier(tol=1e-2, solver='lsqr')),
#		('Perceptron', build_model(Perceptron, 'Classifier', 'Perceptron', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np)),
#		('Passive-Aggressive', PassiveAggressiveClassifier(n_iter=50, n_jobs=1 if opts.mltl else opts.np)),
#		('kNN', KNeighborsClassifier(n_neighbors=100, n_jobs=1 if opts.mltl else opts.np)),
#		('NearestCentroid', NearestCentroid()),
#		('BernoulliNB', BernoulliNB()),
#		('MultinomialNB', MultinomialNB()),
#		('ExtraTrees', build_model(ExtraTreesClassifier, 'Classifier', 'Extra Trees', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=opts.np)),
		('RandomForest', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0)),
#		('RandomForest', Pipeline([('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, n_jobs=opts.np, random_state=0))])),
#		('BaggingkNN', BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, n_jobs=1 if opts.mltl else opts.np, random_state=0)),
#		('BaggingLinearSVC', build_model(BaggingClassifier, 'Classifier', 'Bagging LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, base_estimator=build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False), n_jobs=1 if opts.mltl else opts.np, random_state=0)(LinearSVC(), max_samples=0.5, max_features=0.5)),
#		('LinSVM', build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False)),
		('RbfSVM', build_model(SVC, 'Classifier', 'RBF SVM', tuned=tuned, pr=pr, mltl=opts.mltl))
	]:
		yield clf_name, clf
		clf_names.append(clf_name)
	if (len(glb_clfnames) < len(clf_names)):
		del glb_clfnames[:]
		glb_clfnames.extend(clf_names)
		

# Benchmark Models
def gen_bm_models(tuned=False, glb_filtnames=[], glb_clfnames=[]):
	# Feature Filtering Model
	for filt_name, filter in gen_featfilt(tuned, glb_filtnames):
	# Classification Model
		for clf_name, clf in gen_clfs(tuned, glb_clfnames):
			yield filt_name, filter, clf_name, clf
			del clf
		del filter
		
	
# Combined Models	
def gen_cb_models(tuned=False, glb_filtnames=[], glb_clfnames=[]):
	tuned = tuned or opts.best
	common_cfg = cfgr('chm_annot', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
#	filtref_func = ftslct.filtref(os.path.join(spdr.DATA_PATH, 'X.npz'), os.path.join(spdr.DATA_PATH, 'union_filt_X.npz'))
	for mdl_name, mdl in [
		# ('RandomForest', Pipeline([('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0))])),
		('UDT-RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.decision_tree, k=500, fn=100)), ('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0))])),
		# ('RandomForest', Pipeline([('featfilt', SelectFromModel(DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=0))), ('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np, random_state=0))])),
		# ('RbfSVM102-2', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM 102-2', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('RbfSVM103-2', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM 103-2', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('RbfSVM102-3', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM 102-3', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('RbfSVM103-3', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM 103-3', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('DF-RbfSVM', Pipeline([('featfilt', ftslct.MSelectOverValue(ftslct.filtref(os.path.join(spdr.DATA_PATH, 'X.npz'), os.path.join(spdr.DATA_PATH, 'union_filt_X.npz'), os.path.join(spdr.DATA_PATH, 'orig_X.npz')))), ('clf', build_model(SVC, 'Classifier', 'RBF SVM', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		('RbfSVM', Pipeline([('clf', build_model(SVC, 'Classifier', 'RBF SVM', tuned=tuned, pr=pr, mltl=opts.mltl, probability=True))])),
		# ('L1-LinSVC', Pipeline([('clf', build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False))])),
		# ('Perceptron', Pipeline([('clf', build_model(Perceptron, 'Classifier', 'Perceptron', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=1 if opts.mltl else opts.np))])),
		# ('MNB', Pipeline([('clf', build_model(MultinomialNB, 'Classifier', 'MultinomialNB', tuned=tuned, pr=pr, mltl=opts.mltl))])),
#		('5NN', Pipeline([('clf', build_model(KNeighborsClassifier, 'Classifier', 'kNN', tuned=tuned, pr=pr, mltl=opts.mltl, n_neighbors=5, n_jobs=1 if opts.mltl else opts.np))])),
		# ('MEM', Pipeline([('clf', build_model(LogisticRegression, 'Classifier', 'Logistic Regression', tuned=tuned, pr=pr, mltl=opts.mltl, dual=False))])),
		# ('LinearSVC with L2 penalty [Ft Filt] & Perceptron [CLF]', Pipeline([('featfilt', SelectFromModel(build_model(LinearSVC, 'Feature Selection', 'LinearSVC', tuned=tuned, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False, penalty='l2'))), ('clf', build_model(Perceptron, 'Classifier', 'Perceptron', tuned=tuned, pr=pr, n_jobs=opts.np))])),
		# ('ExtraTrees', Pipeline([('clf', build_model(ExtraTreesClassifier, 'Classifier', 'Extra Trees', tuned=tuned, pr=pr, mltl=opts.mltl, n_jobs=opts.np))])),
#		('Random Forest', Pipeline([('clf', build_model(RandomForestClassifier, 'Classifier', 'Random Forest', tuned=tuned, pr=pr, n_jobs=opts.np, random_state=0))]))
	]:
		yield mdl_name, mdl


# Models with parameter range
def gen_mdl_params(rdtune=False):
	common_cfg = cfgr('chm_annot', 'common')
	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
	if (rdtune):
		for mdl_name, mdl, params in [
			# ('Logistic Regression', LogisticRegression(dual=False), {
				# 'param_dist':dict(
					# penalty=['l1', 'l2'],
					# C=np.logspace(-5, 5, 11),
					# tol=np.logspace(-6, 3, 10)),
				# 'n_iter':30
			# }),
			# ('LinearSVC', LinearSVC(dual=False), {
				# 'param_dist':dict(
					# penalty=['l1', 'l2'],
					# C=np.logspace(-5, 5, 11),
					# tol=np.logspace(-6, 3, 10)),
				# 'n_iter':30
			# }),
			# ('Perceptron', Perceptron(), {
				# 'param_dist':dict(
					# alpha=np.logspace(-6, 3, 10),
					# n_iter=stats.randint(3, 20)),
				# 'n_iter':30
			# }),
			# ('MultinomialNB', MultinomialNB(), {
				# 'param_dist':dict(
					# alpha=np.logspace(-6, 3, 10),
					# fit_prior=[True, False]),
				# 'n_iter':30
			# }),
			# ('SVM', SVC(), {
				# 'param_dist':dict(
					# kernel=['linear', 'rbf', 'poly'],
					# C=np.logspace(-5, 5, 11),
					# gamma=np.logspace(-6, 3, 10)),
				# 'n_iter':30
			# }),
			# ('Extra Trees', ExtraTreesClassifier(random_state=0), {
				# 'param_dist':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					# min_samples_leaf=[1]+range(10, 101, 10),
					# class_weight=['balanced', None]),
				# 'n_iter':30
			# }),
			('Random Forest', RandomForestClassifier(random_state=0), {
				'param_dist':dict(
					n_estimators=[50, 100] + range(200, 1001, 200),
					max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					max_depth=[None] + range(10,101,10),
					min_samples_leaf=[1]+range(10, 101, 10),
					class_weight=['balanced', None]),
				'n_iter':30
			}),
			# ('Bagging LinearSVC', BaggingClassifier(base_estimator=build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=opts.best, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False), random_state=0), {
				# 'param_dist':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# max_samples=np.linspace(0.5, 1, 6),
					# max_features=np.linspace(0.5, 1, 6),
					# bootstrap=[True, False],
					# bootstrap_features=[True, False]),
				# 'n_iter':30
			# }),
			# ('AdaBoost LinearSVC', AdaBoostClassifier(base_estimator=build_model(SVC, 'Classifier', 'SVM', tuned=opts.best, pr=pr, mltl=opts.mltl), algorithm='SAMME', random_state=0), {
				# 'param_dist':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# learning_rate=np.linspace(0.5, 1, 6)),
				# 'n_iter':30
			# }),
			# ('GB LinearSVC', GradientBoostingClassifier(random_state=0), {
				# 'param_dist':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# subsample = np.linspace(0.5, 1, 6),
					# max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					# min_samples_leaf=[1]+range(10, 101, 10),
					# learning_rate=np.linspace(0.5, 1, 6),
					# loss=['deviance', 'exponential']),
				# 'n_iter':30
			# }),
			# ('UGSS & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.gss_coef, fn=4000)), ('clf', RandomForestClassifier())]), {
				# 'param_dist':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int')),
				# 'n_iter':8
			# }),
		]:
			yield mdl_name, mdl, params
	else:
		for mdl_name, mdl, params in [
			# ('Logistic Regression', LogisticRegression(dual=False), {
				# 'param_grid':dict(
					# penalty=['l1', 'l2'],
					# C=np.logspace(-5, 5, 11),
					# tol=np.logspace(-6, 3, 10))
			# }),
			# ('LinearSVC', LinearSVC(dual=False), {
				# 'param_grid':dict(
					# penalty=['l1', 'l2'],
					# C=np.logspace(-5, 5, 11),
					# tol=np.logspace(-6, 3, 10))
			# }),
			# ('Perceptron', Perceptron(), {
				# 'param_grid':dict(
					# alpha =np.logspace(-5, 5, 11),
					# n_iter=range(3, 20, 3))
			# }),
			# ('MultinomialNB', MultinomialNB(), {
				# 'param_grid':dict(
					# alpha=np.logspace(-6, 3, 10),
					# fit_prior=[True, False])
			# }),
			# ('SVM', SVC(), {
				# 'param_grid':dict(
					# kernel=['linear', 'rbf', 'poly'],
					# C=np.logspace(-5, 5, 11),
					# gamma=np.logspace(-6, 3, 10))
			# }),
			# ('Extra Trees', ExtraTreesClassifier(random_state=0), {
				# 'param_grid':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					# min_samples_leaf=[1]+range(10, 101, 10),
					# class_weight=['balanced', None])
			# }),
			('Random Forest', RandomForestClassifier(random_state=0), {
				'param_grid':dict(
					n_estimators=[50, 100] + range(200, 1001, 200),
					max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					max_depth=[None] + range(10,101,10),
					min_samples_leaf=[1]+range(10, 101, 10),
					class_weight=['balanced', None])
			}),
			# ('Bagging LinearSVC', BaggingClassifier(base_estimator=build_model(LinearSVC, 'Classifier', 'LinearSVC', tuned=opts.best, pr=pr, mltl=opts.mltl, loss='squared_hinge', dual=False), random_state=0), {
				# 'param_grid':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# max_samples=np.linspace(0.5, 1, 6),
					# max_features=np.linspace(0.5, 1, 6),
					# bootstrap=[True, False],
					# bootstrap_features=[True, False])
			# }),
			# ('AdaBoost LinearSVC', AdaBoostClassifier(base_estimator=build_model(SVC, 'Classifier', 'SVM', tuned=opts.best, pr=pr, mltl=opts.mltl), algorithm='SAMME', random_state=0), {
				# 'param_grid':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# learning_rate=np.linspace(0.5, 1, 6))
			# }),
			# ('GB LinearSVC', GradientBoostingClassifier(random_state=0), {
				# 'param_grid':dict(
					# n_estimators=[50, 100] + range(200, 1001, 200),
					# subsample = np.linspace(0.5, 1, 6),
					# max_features=np.linspace(0.5, 1, 6).tolist()+['sqrt', 'log2'],
					# min_samples_leaf=[1]+range(10, 101, 10),
					# learning_rate = np.linspace(0.5, 1, 6),
					# loss=['deviance', 'exponential'])
			# }),
			# ('UDT & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.decision_tree, fn=4000)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('DT & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.decision_tree)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('UNGL & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.ngl_coef, fn=4000)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('NGL & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.ngl_coef)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('UGSS & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.utopk, filtfunc=ftslct.gss_coef, fn=4000)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# }),
			# ('GSS & RF', Pipeline([('featfilt', ftslct.MSelectKBest(ftslct.gss_coef)), ('clf', RandomForestClassifier())]), {
				# 'param_grid':dict(
					# featfilt__k=np.logspace(np.log2(250), np.log2(32000), 8, base=2).astype('int'))
			# })
		]:
			yield mdl_name, mdl, params


def all():
	global FILT_NAMES, CLF_NAMES, PL_NAMES, PL_SET, cfgr

	if (opts.mltl):
		pid = 'all'
	else:
		pid = opts.pid
	print 'Process ID: %s' % pid
	
	## Load data
	# From text
#	pmid_list = spdr.get_pmids()
#	articles = spdr.fetch_artcls(pmid_list)
#	X, Y = spdr.get_data(articles)
	X, Y = load_data(opts.mltl, pid, opts.spfmt)
	
	## Cross validation
	model_iter = gen_cb_models if opts.comb else gen_bm_models
	model_param = dict(tuned=opts.best, glb_filtnames=FILT_NAMES, glb_clfnames=CLF_NAMES)
	global_param = dict(comb=opts.comb, pl_names=PL_NAMES, pl_set=PL_SET)
	txtclf.cross_validate(X, Y, model_iter, model_param, avg=opts.avg, kfold=opts.kfold, cfg_param=cfgr('bionlp.txtclf', 'cross_validate'), global_param=global_param, lbid=pid)

	
def from_pbmd():
#	pmid_list = pm.get_pmids('Proliferation Receptor Cancer', 10)
#	pmid_list = ['1280402']
#	articles = pm.fetch_artcls(pmid_list)
	pass

	
def tuning():
	from sklearn.model_selection import KFold
	if (opts.mltl):
		pid = 'all'
	else:
		pid = opts.pid
	print 'Process ID: %s' % pid
	
	## Load data
	X, y = load_data(opts.mltl, pid, opts.spfmt)
	X = X.as_matrix()
	
	## Parameter tuning
	print 'Parameter tuning is starting...'
	ext_params = dict(cv=KFold(n_splits=opts.kfold, shuffle=True, random_state=0))
	for mdl_name, mdl, params in gen_mdl_params(opts.rdtune):
		params.update(ext_params)
		print 'Tuning hyperparameters for %s' % mdl_name
		pt_result = txtclf.tune_param(mdl_name, mdl, X, y, opts.rdtune, params, mltl=opts.mltl, avg='micro' if opts.avg == 'all' else opts.avg, n_jobs=opts.np)
		io.write_npz(dict(zip(['best_params', 'best_score', 'score_avg_cube', 'score_std_cube', 'dim_names', 'dim_vals'], pt_result)), '%sparam_tuning_for_%s_%s' % ('rd_' if opts.rdtune else '', mdl_name.replace(' ', '_').lower(), pid))


def demo():
	import urllib
	global cfgr
	if not os.path.exists('data'):
		os.makedirs('data')
	urllib.urlretrieve ('http://data.mendeley.com/datasets/s9m6tzcv9d/3/files/87afede7-5a4c-4cee-99d3-45cc638b5d12/udt_exp_X.npz', 'data/X.npz')
	urllib.urlretrieve ('http://data.mendeley.com/datasets/s9m6tzcv9d/3/files/bfb90278-c313-47c6-ace7-2dd1e48b5daa/Y.npz', 'data/Y.npz')
	hoc.DATA_PATH = 'data'
	X, Y = load_data(True, 'all', opts.spfmt)
	def model_iter(tuned, glb_filtnames, glb_clfnames):
		yield 'UDT-RF', Pipeline([('clf', OneVsRestClassifier(RandomForestClassifier(max_features=0.7, min_samples_leaf=1, n_estimators=200, class_weight='balanced'), n_jobs=opts.np))])
	txtclf.cross_validate(X, Y, model_iter, model_param=dict(tuned=False, glb_filtnames=[], glb_clfnames=[]), avg='micro', kfold=5, cfg_param=cfgr('bionlp.txtclf', 'cross_validate'), global_param=dict(comb=True, pl_names=[], pl_set=set([])), lbid=-1)


def main():
	if (opts.tune):
		tuning()
		return
	if (opts.method == 'demo'):
		demo()
		return
	all()


if __name__ == '__main__':
	# Logging setting
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

	# Parse commandline arguments
	op = OptionParser()
	op.add_option('-k', '--kfold', default=10, action='store', type='int', dest='kfold', help='indicate the K fold cross validation')
	op.add_option('-p', '--pid', default=0, action='store', type='int', dest='pid', help='indicate the process ID')
	op.add_option('-n', '--np', default=-1, action='store', type='int', dest='np', help='indicate the number of processes used for training')
	op.add_option('-f', '--fmt', default='npz', help='data stored format: csv or npz [default: %default]')
	op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csr or csc [default: %default]')
	op.add_option('-t', '--tune', action='store_true', dest='tune', default=False, help='firstly tune the hyperparameters')
	op.add_option('-r', '--rdtune', action='store_true', dest='rdtune', default=False, help='randomly tune the hyperparameters')
	op.add_option('-b', '--best', action='store_true', dest='best', default=False, help='use the tuned hyperparameters')
	op.add_option('-c', '--comb', action='store_true', dest='comb', default=False, help='run the combined methods')
	op.add_option('-l', '--mltl', action='store_true', dest='mltl', default=False, help='use multilabel strategy')
	op.add_option('-a', '--avg', default='micro', help='averaging strategy for performance metrics: micro or macro [default: %default]')
	op.add_option('-i', '--input', default='hoc', help='input source: hoc or pbmd [default: %default]')
	op.add_option('-m', '--method', help='main method to run')

	(opts, args) = op.parse_args()
	if len(args) > 0:
		op.print_help()
		op.error('Please input options instead of arguments.')
		exit(1)
		
	spdr = SPDR_MAP[opts.input]
	# Parse config file
	if (os.path.exists(CONFIG_FILE)):
		cfgr = io.cfg_reader(CONFIG_FILE)
		spdr_cfg = cfgr('bionlp.spider.%s' % opts.input, 'init')
		if (len(spdr_cfg) > 0 and spdr_cfg['DATA_PATH'] is not None and os.path.exists(spdr_cfg['DATA_PATH'])):
			spdr.DATA_PATH = spdr_cfg['DATA_PATH']
		plot_cfg = cfgr('bionlp.util.plot', 'init')
		plot_common = cfgr('bionlp.util.plot', 'common')
		txtclf.init(plot_cfg=plot_cfg, plot_common=plot_common)

	main()