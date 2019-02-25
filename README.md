# cvLME

<h3>A multi-language library to perform cross-validated Bayesian model selection</h3>

This package allows to calculate the <i>cross-validated log model evidence</i> (cvLME) [1,2,3] in multiple programming languages.

Based on calculated cvLMEs, <i>cross-validated Bayesian model selection</i> (cvBMS) within a model space can be performed.

Currently, it supports the following model structures

- `MS` = model space; for general model selection operations;
- `GLM` = univariate general linear model; for linear regression;
- `Poiss` = Poisson distribution with exposures; for count data;

which are implemented in the following languages

- `LaTeX`: for documentation purposes only, written in TeXstudio 2.8.8; 
- `MATLAB`: developed in and compatible with MATLAB R2013b;
- `Python`: developed in and compatible with Python 3.7.

Extensive documentation is given in the manual accompanying this repository [4].

In the following examples, `<name-of-the-model-class>` is either `"GLM"` or `"Poiss"`.


<br>
<h3>Getting started with Python</h3>

To use the module, it is simply imported via `import cvBMS`, e.g. at the beginning of your analysis script.

In a Python console, type `help(cvBMS)` and `help(cvBMS.<name-of-the-model-class>)` to learn more.

Please also read the implementation notes in `LaTeX\cvBMS.pdf` [4] to apply in Python.


<br>
<h3>Getting started with MATLAB</h3>

To use these functions, simply rename and put the sub-directory `MATLAB` into your MATLAB path.

In the command window, type `help <name-of-the-model-class>_cvLME.m` to learn more.

Please also read the implementation notes in `LaTeX\cvBMS.pdf` [4] to apply in MATLAB.


<br>
<h3>Getting started with LaTeX</h3>

Simply open `<name-of-the-model-class>.tex` in sub-directory `LaTeX` to access and reuse formulas.

Please open `LaTeX\cvBMS.pdf` [4] to view the PDF output from this LaTeX code.


<br>
<h3>References</h3>

[1] https://www.sciencedirect.com/science/article/pii/S1053811916303615 <br>
[2] https://www.sciencedirect.com/science/article/pii/S105381191730527X <br>
[3] https://www.sciencedirect.com/science/article/pii/S0165027018301468 <br>
[4] https://github.com/JoramSoch/cvLME/blob/master/LaTeX/cvBMS.pdf <br>
