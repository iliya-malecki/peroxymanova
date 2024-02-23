# Peroxymanova

This project is essentially an implementation of PERMANOVA ([wiki link](https://en.wikipedia.org/wiki/Permutational_analysis_of_variance), doi:[10.1111/j.1442-9993.2001.01070.pp.x](https://onlinelibrary.wiley.com/doi/10.1111/j.1442-9993.2001.01070.pp.x)) in Rust.

PERMANOVA (Permutational Multivariate Analysis of Variance) is a method for comparing groups of mathematical objects, requiring only a dissimilarity matrix between them, as opposed to having a notion of an average, like the one used in classical ANOVA. This is incredibly useful, since it is massively easier to define a dissimilarity than a mean: there is no obvious "average graph", "average neural network" or an "average RL policy", but with a little bit of hand waving one can define distances, or dissimilarities between a pair of such entities.

This package aims to provide quality-of-life bells and whistles that turn this incredible method into something useful day to day. It implements the following workflow:

1. Accept a `Collection` of some things `T`, a `Callable` that can compare two of those `T`, returning a float, and a `Collection` of labels that indicate to which group a given thing `T` belongs to
2. Efficiently run the `Callable[[T,T], float]` for every possible pair of objects in the `Collection` and build a dissimilarity matrix
3. Given the dissimilarity matrix and a `Collection` of group-indicating labels, run the PERMANOVA algorithm to get a test statistic and a p-value for the null hypothesis of the groups being all and the same. This step requires a lot of permutations to get the p-value so run it *blazingly fast* in Rust

Strategic roadmap:
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
