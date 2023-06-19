<h1 align="center">
<br> Characterizing Branching Processes from Sampled Data
</h1>
Repository for the course on <a href="https://www.cos.ufrj.br/~daniel/mcmc/">Monte Carlo Algorithms and Markov Chains</a> at  <a href="https://www.cos.ufrj.br/" > PESC - Programa de Engenharia de Sistemas e Computação</a> from <a href="https://ufrj.br/" >UFRJ - Federal University of Rio de Janeiro</a>, taught by <a href="https://www.cos.ufrj.br/~daniel/">Prof.  Daniel Ratton Figueiredo</a>.

Developed by Ronald Albert.
<h2 align="center">
The project
</h2>
The project is an implementation of the approach taken by the <a href="https://arxiv.org/abs/1302.5847">article</a> from Fabricio Murai, Bruno Ribeiro, Don Towsley e Krista Gile, where the authors estimate the offspring distribution of a branching process from observed nodes of the generated tree. <strong>This project is a mere implementation of the methodology developed by the authors of the article mentioned above.</strong>

It's entirely implemented in python and requires several python libraries to be executed. All of the required libraries are listed at [requirements.txt](requirements.txt) as well as their respective versions. In order to install all the necessary package one could run the following command
```
pip install -r requirements.txt
```

<h2 align="center">
File list
</h2>
<ul>
    <li><h3>run.py</h3></li>
    <p>Script that runs the entire project. It's the main file of the project, and generates the results available at the results folder.</p>
    <li><h3>experiment.py</h3></li>
    <p>Script where the function for a single experiment is defined as well as the evaluation function for the results obtained.</p>
    <li><h3>sampling_model.py</h3></li>
    <p>Script for functions related to sampling and doing inference on Galton-Watson processes. They are mostly used as auxiliary function for the MCMC algorithm.</p>
    <li><h3>graph_mcmc.py</h3></li>
    <p>Script where the MCMC algorithm is defined, the transition and acceptance probability functions serve as auxiliary for the main function of generating a path in the Markov Chain constructed.</p>
    <li><h3>optimization.py</h3></li>
    <p>Script for the optimization fo the likelihood function in order to estimate the offspring distribution of the desired Branching Proccess.</p>
</ul>

<h2 align="center">
Execution
</h2>
<p>After installing all the necessary packages, one can run the project by executing the following command</p>

```
python run.py
```

<h2 align="center">
Results
</h2>
<p>The results obtained by running the project are available at the results folder. The results are in pickle format and can de loaded into a python Dictionary object by perfoming the command</p>

```
with open('results/<result_file_name>.pkl', 'rb') as f:
    results = pickle.load(f)
```
