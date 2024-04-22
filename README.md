# Peroxymanova

This package allows you to measure differences between groups of things, just like ANOVA (or have you heard of t-test? A/B testing?) - but without needing those things to be numbers! Instead, it wants you to provide a function to compare two objects, and does the rest.

## Quickstart

```bash
pip install peroxymanova
```
And you're good to go!

You can also install it from github directly, but that requires having Rust toolchain since it involves building from source. Since this is a small project, there is no release schedule and changes get released pretty much immediately, so there is no reason to install from github, building from source.

```python
from peroxymanova import permanova_pipeline
permanova_pipeline(...) # open the suggestions and follow the types!
```

## Description

This project is essentially an implementation of PERMANOVA ([wiki](https://en.wikipedia.org/wiki/Permutational_analysis_of_variance)) in Rust.

PERMANOVA (Permutational Multivariate Analysis of Variance) is a method for comparing groups of mathematical objects, requiring only a dissimilarity matrix between them, as opposed to having a notion of an average, like the one used in classical ANOVA. This is incredibly useful, since it is massively easier to define a dissimilarity than a mean: there is no obvious "average graph", "average neural network" or an "average RL policy", but with a little bit of hand waving one can define distances, or dissimilarities between a pair of such entities.

In practice, it means that if you have groups of object you dont know how to represent in numbers, but you need to compare them, you are out of luck with ANOVA - but you can use this package instead.

This package aims to provide quality-of-life bells and whistles that turn PERMANOVA into something useful day to day. It implements the following workflow:

1. Accept a set of some things `T`, a `Callable` that can compare two of those `T`s, returning a `float`, and an array of labels that indicate which group a given thing `T` belongs to
2. Efficiently run the `Callable[[T,T], float]` for every possible pair of things and build a dissimilarity matrix
3. Run the PERMANOVA algorithm to get a test statistic and a p-value for the null hypothesis of the groups being all and the same. This step requires a lot of permutations to get the p-value so run it *blazingly fast* in Rust


## Example problems (what does this package do again?)


### League of Legends bot (or any other multi-objective genetic optimization)

Imagine you are building a bot using genetic RL. The name of the game in RL is the ability to explore the whole parameter space while being able to fine tune well - especially if there are multiple things to do at once, that is hard. A valid approach for achieving this is getting a PhD in RL. However, another (maybe less) valid approach is to split your agent generations into niches and let them fine tune their parameters, while keeping the niches far apart in the parameter space to keep exploring parameters well. Of course, getting a PhD in RL will also come with the understanding of how to do this, but another way is to use ANOVA. There is a caveat: it is often impossible to fully represent an agent as an R^d, so in those cases PERMANOVA comes to the rescue! Run your niches through the algorithm once to get yet another number for maximization purposes, or get a p-value approximation if you have a heuristic for the sweet spot for the distance between your groups. (A note for clarity: of course, you can do niching withot this package, this is one of many ways it can be done)

### Mouse movement analysis (or any other graph problem)

Imagine you're setting up an experiment about mice (the squeaky mammals, not computer mice) solving mazes (this actually happened and is the inspiration for this project). You're reprezenting the mouse movement in the maze as a directed graph. You're doing something to the mice and measuring how they solve mazes with different experimental factors, but for some complicated reason you cant phrase your analysis as a simple repeated measures ANOVA / paired t-test (maybe the mice are always different!). You want to run an ANOVA but each measurement is a directed graph! Representing a graph as an R^d using graph measures is a terrible idea since it loses a lot of information - a better way is to define a distance between graphs that encompases the changes you wanted to measure all along. This way, with a matrix of distances between graphs, you can run PERMANOVA and get your answers!

### String comparisons (or you arent a DS person and the examples above dont make you excited)

Imagine you have two groups of strings you need to compare. Maybe they are satled and unsalted password hashes and youre checking your salting. There is literally no way to represent the passwords as a float or an array of floats (what i was calling R^d for ML/DS people :) ). However, you can compare strings with things like the Levenshtein distance! So here PERMANOVA definitely comes to the rescue, you just run the thing with levenshtein distance and get your p-value that tells you the probability your salted passwords dont look different from unsalted ones based on used characters.

### Binary blob comparisons (or you like pushing tech too far)

Imagine you have several sets of binary blobs. Maybe they are binary messages you were sharing with your friends Alice and Bob, and each set is, as you suspect, a repeated message. Run the PERMANOVA on the hamming distances between them to know fur sure! This is because in science everything with p-value < 0.05 is true, and otherwise false (that was a joke, please always remember the fraction of experiments that would lead to accepting the null hypothesis is right there in the p-value).


## Further reading

please take a look at the [wiki](https://en.wikipedia.org/wiki/Permutational_analysis_of_variance) page and the original paper: [doi](https://onlinelibrary.wiley.com/doi/10.1111/j.1442-9993.2001.01070.pp.x)

## Strategic roadmap

- Actually make it multivariate since a single p-value for a single "difference" is kind of just one-way permutational ANOVA.
- Make a fancy parallelization backend interface for computing pairwise distances. Maybe there could be a backend='ray' that would actually search for a full ray cluster?
- Since we dream of ray, should we get a cluster for rust side as well? :)

## Development
### Releasing

Since were using pyproject.toml, the source of truth for the version is the file,
and the version tag is based on the `project.version` attribute. It means
bumping the version necessitates a separate commit.
At least there is a way to ensure the version tag consistency with the version
attribute in the file. Use:
```bash
git config --local include.path ../.gitconfig
```
to add the repos' .gitconfig with an ugly but nifty "git bump" alias
