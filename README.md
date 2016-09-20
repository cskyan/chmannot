# Cancer Hallmark Annotator

`chmannot` is an automatic annotator of cancer hallmark on biomedical literature. It supports abstract-level annotation which means that given the abstract of a paper in PubMed it could predict the hallmark labels related to this paper. It is mainly used to evaluate the computational models and tune the model parameters. It also provides several utility functions to manipulate the dataset and post-process the results.

## Getting Started

The following instructions will help you get a copy of the source code as well as the datasets, and run the programs on your own machine.

### Prerequisities

Firstly, you need to install a Python Interpreter (tested 2.7.12) and these packages:

* numpy (tested 1.11.1)
* scipy (tested 1.11.1)
* matplotlib (tested 1.5.1)
* pandas (tested 0.18.1)
* scikit-learn (tested 0.17.1)
* pyyaml (test 3.11)
* openpyxl (test 2.3.2)
* rdflib \[optional\] \(tested 4.2.1\)

The simplest way to get started is to use [Anaconda](https://www.continuum.io/anaconda-overview) Python distribution. If you have limited disk space, the [Miniconda](http://conda.pydata.org/miniconda.html) installer is recommended. After installing Miniconda and adding the path of folder `bin` to `$PATH` variable, run the following command:

```bash
conda install scikit-learn pandas matplotlib openpyxl
```

### Download the Source Code

You can clone the repository of this project and then update the submodule after entering the main folder:

```bash
git clone https://github.com/cskyan/chmannot.git
cd hmannot
git submodule update --init --recursive
```

Or you can clone the repository and submodules simultaneously:

```bash
git clone --recursive https://github.com/cskyan/chmannot.git
```

### Configure Environment Variable

* Add the path of folder `bin` to `$PATH` variable so that you can run the scripts wherever you want. *Remember to grant execution permissions to all the files located in* `bin`
* Add the path of folder `lib` to `$PYTHONPATH` variable so that the Python Interpreter can find the library `bionlp`.

### Configuration File

The global configuration file is stored as `etc/config.yaml`. The configurations of different functions in different modules are separated, which looks like the code snippet below.

```
MODULE1:
- function: FUNCTION1
  params:
    PARAMETER1: VALUE1
    PARAMETER2: VALUE2
- function: FUNCTION2
  params:
    PARAMETER1: VALUE1
	
MODULE2:
- function: FUNCTION1
  params:
    PARAMETER1: VALUE1
```

Hence you can access a specific parameter VALUE using a triple (MODULE, FUNCTION, PARAMETER). The utility function `cfg_reader` in `bionlp.util.io` can be used to read the parameters in the configuration file:

```python
import bionlp.util.io as io
cfgr = io.cfg_reader(CONFIG_FILE_PATH)
cfg = cfgr(MODULE, FUNCTION)
VALUE = cfg[PARAMETER]
```

The parameters under the function `init` means that they are defined in module scope, while the parameters under the function `common` means that they are shared among all the functions inside the corresponding module.

### Locate the Pre-Generated Dataset

After cloning the repository, you can download some pre-generated datasets [here](https://data.mendeley.com/datasets/s9m6tzcv9d) . The datasets described below are organized as [csc sparse matrices](http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html), stored in compressed `npz` files using the [function](http://docs.scipy.org/doc/numpy/reference/generated/numpy.savez_compressed.html) of `numpy`. 

Filename | Description  
--- | ---  
orig_X.npz | Standard dataset
exp_X.npz | Expanded dataset
udt_orig_X.npz | Standard dataset filtered by UDT
udt_exp_X.npz | Expanded dataset filtered by UDT
dt_orig_X.npz | Standard dataset filtered by DT
dt_exp_X.npz | Expanded dataset filtered by DT
union_filt_X.npz | Standard dataset filtered by DF
X_[0-9].npz | Separated standard dataset
Y.npz | Cancer hallmark labels
y_[0-9].npz | Separated cancer hallmark label

**In order to locate the dataset you want to use, please rename it to 'X.npz', and change the parameter `DATA_PATH` of module `bionlp.spider.hoc` inside `etc/config.yaml` into the location of 'X.npz'.**

You can load a dataset into a [Pandas DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html), with the corresponding `PMID` as index and each feature as column name, using the utility function `read_df` in `bionlp.util.io`:

```python
import bionlp.util.io as io
X = io.read_df('X.npz', with_idx=True, sparse_fmt='csc')
```

### A Simple Example

You can run a demo using the following command:

```bash
chm_annot.py -m demo
```

*If your operating system is Windows, please use the Python Interpreter to execute the python scripts:*

```bash
python chm_annot.py -m demo
```

This demo will automatically download a dataset and perform a 5-fold cross validation on the proposed method UDT-RF. The log is printed to standard output and the results are saved on the disk.

## Parameter Tuning

For the sake of the best performance, you should tune the parameters of your selected model and write them on the model configuration file so that you can use these tuned parameters for model evaluation. 

### Setup parameter range

You can edit the function `gen_mdl_params` inside `bin/chm_annot.py` to change the range of parameter tuning. Please uncomment the code lines corresponded to your selected model and change the range of the parameters or append other values you want to test.

### Run parameter tuning script

You can choose an approach for parameter tuning using the following command.

*Grid Search*:

```bash
chm_annot.py -t
```

*Random Search*:

```bash
chm_annot.py -t -r
```

### Covert the result to configuration file

You can use the utility function in `bin/chm_helper.py` to transformat your tuning result by the following command:

```bash
chm_helper.py -m n2y -l TUNING_OUTPUT_FOLDER_PATH
```

**Then copy the basename of the configuration file ended with `.yaml` to the parameter `mdl_cfg` of module `chm_annot` inside `etc/config.yaml`.**

The pre-tuned parameters for some models are stored in `etc/mdlcfg.yaml`.

## Model Evaluation

You can use different combination of the feature selection model and classification model to generate a pipeline as the final computational model.

You can uncomment the corresponding code lines of the models you want to evaluate in function `gen_featfilt` and `gen_clfs` inside `bin/chm_annot.py` for feature selection and classification respectively.

In addition, you can use command line parameter `-c` to adopt the pre-combined model in function `gen_cb_models`. To make use of the parameters stored in configuration file, you can use command line parameter `-c -b` to adopt the pre-combined model with optimized parameters.

## Dataset Re-Generation

You can re-generate the dataset from the [pre-processed files](http://www.cl.cam.ac.uk/~sb895/HoC.html) stored in `DATA_PATH` using the following command:

```bash
chm_gendata.py -m gen
```

It will also generate separated label data `y_[0-9].npz` for single label running.

Feature selection method can also be applied to the dataset in advance by uncommenting the corresponding code line in function `gen_data` inside `bin\chm_gendata.py`.

If you only want to apply feature selection on the generated dataset or generate separated label data, you can use command line parameter `-l`. Make sure your dataset has already been renamed as 'X.npz' and the processed dataset will be generated as 'new_X.npz'.

## Common Parameter Setting

* _-p [0-9]_  
specify which label you want to use independently
* _-l_  
indicate that you want to use all labels simultaneously
* _-k NUM_  
specify *K*-fold cross validation
* _-a [micro | macro]_  
specify which average strategy you want to use for multi-label annotation
* _-n NUM_  
specify how many CPU cores you want to use simultaneously for parallel computing

**Other parameter specification can be obtained using `-h`.**