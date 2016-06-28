import operator
from copy import copy
from hashlib import sha1
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import chisqprob
import Orange
from Orange.classification import Learner, Model


def argmaxrnd(a, random_seed=None):
    """
    Returns the index of the maximum value for a given 1D array. In case
    of multiple indices corresponding to the maximum value, the result
    is chosen randomly among those. The random number generator can be
    seeded by forwarding a seed value: see function 'hash_dist'.

    Multidimensional arrays are also supported.

    :param x: 1D input array (vector) of real numbers
    :type x: np.ndarray
    :param random_seed: used to initialize the random number generator
    :type random_seed: int
    :return: index of the maximum value
    """
    random = (np.random if random_seed is None
              else np.random.RandomState(random_seed))

    def f(x): return random.choice((x == np.nanmax(x)).nonzero()[0])
    return f(a) if a.ndim == 1 else np.apply_along_axis(f, axis=1, arr=a)


def entropy(x):
    x = x[x != 0]
    x = x / np.sum(x, axis=0)
    x = -x * np.log2(x)
    return np.sum(x)


def likelihood_ratio_statistic(x, y):
    x[x == 0] = 1e-5
    y[y == 0] = 1e-5
    lrs = np.sum(x * np.log(x / y))
    lrs = 2 * (lrs - np.sum(x) * np.log(np.sum(x) / np.sum(y)))
    return lrs


def get_dist(Y, domain):
    return np.bincount(Y.astype(dtype=np.int),
                       minlength=len(domain.class_var.values))


def hash_dist(x):
    return int(sha1(bytes(x)).hexdigest(), base=16) & 0xffffffff


def rule_length(rule):
    return len(rule.selectors)


class Evaluator:
    def evaluate_rule(self, rule):
        raise NotImplementedError


class EntropyEvaluator(Evaluator):
    def evaluate_rule(self, rule):
        x = rule.curr_class_dist.astype(dtype=np.float)
        if rule.target_class is not None:
            x = np.array([x[rule.target_class],
                          np.sum(x) - x[rule.target_class]])
        return -entropy(x)


class LaplaceAccuracyEvaluator(Evaluator):
    def evaluate_rule(self, rule):
        if rule.target_class is not None:
            target = rule.curr_class_dist[rule.target_class]
            combined = np.sum(rule.curr_class_dist)
            k = 2
        else:
            target = np.argmax(rule.curr_class_dist)
            combined = np.sum(rule.curr_class_dist)
            k = len(rule.curr_class_dist)
        return (target + 1) / (combined + k)


class LengthEvaluator(Evaluator):
    def evaluate_rule(self, rule):
        return -rule_length(rule)


class Validator:
    def validate_rule(self, rule):
        raise NotImplementedError


class CustomGeneralValidator(Validator):
    """
    Discards rules that

    - cover less than the minimum required number of examples,
    - are too complex,
    - or offer no additional advantage in comparison to the parent rule.
    """
    def __init__(self, max_rule_length=5, minimum_covered_examples=1):
        self.max_rule_length = max_rule_length
        self.minimum_covered_examples = minimum_covered_examples

    def validate_rule(self, rule):
        return (rule.num_covered >= self.minimum_covered_examples and
                rule_length(rule) <= self.max_rule_length and
                (rule.num_covered != rule.parent_rule.num_covered
                 if (rule.parent_rule is not None and
                     rule_length(rule.parent_rule) > 0) else True))


class LRSValidator(Validator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def validate_rule(self, rule):
        if self.alpha >= 1.0 or rule.parent_rule is None:
            return True

        x = rule.curr_class_dist.astype(dtype=np.float)
        y = rule.parent_rule.curr_class_dist.astype(dtype=np.float)
        if rule.target_class is not None:
            x = np.array([x[rule.target_class],
                          np.sum(x) - x[rule.target_class]])
            y = np.array([y[rule.target_class],
                          np.sum(y) - y[rule.target_class]])

        lrs = likelihood_ratio_statistic(x, y)
        return (lrs > 0 and
                chisqprob(lrs, len(rule.curr_class_dist) - 1) <= self.alpha)


class SearchAlgorithm:
    def select_candidates(self, rules):
        raise NotImplementedError

    def filter_rules(self, rules):
        raise NotImplementedError


class BeamSearchAlgorithm(SearchAlgorithm):
    def __init__(self, beam_width=5):
        self.beam_width = beam_width

    def select_candidates(self, rules):
        return rules, []

    def filter_rules(self, rules):
        return rules[:self.beam_width]


class SearchStrategy:
    def initialize_rule(self, X, Y, target_class, base_rules, domain,
                        prior_class_dist, quality_evaluator,
                        complexity_evaluator, significance_validator,
                        general_validator):
        raise NotImplementedError

    def refine_rule(self, X, Y, candidate_rule):
        raise NotImplementedError


class TopDownSearch(SearchStrategy):
    def initialize_rule(self, X, Y, target_class, base_rules, domain,
                        prior_class_dist, quality_evaluator,
                        complexity_evaluator, significance_validator,
                        general_validator):
        rules = []
        general_rule = Rule(domain=domain, prior_class_dist=prior_class_dist,
                            quality_evaluator=quality_evaluator,
                            complexity_evaluator=complexity_evaluator,
                            significance_validator=significance_validator,
                            general_validator=general_validator)

        general_rule.filter_and_store(X, Y, target_class)
        rules.append(general_rule)

        for base_rule in base_rules:
            temp_rule = Rule(selectors=copy(base_rule.selectors),
                             parent_rule=general_rule, domain=domain,
                             prior_class_dist=prior_class_dist,
                             quality_evaluator=quality_evaluator,
                             complexity_evaluator=complexity_evaluator,
                             significance_validator=significance_validator,
                             general_validator=general_validator)

            temp_rule.filter_and_store(X, Y, target_class)
            if temp_rule.validity:
                rules.append(temp_rule)

        return rules

    def refine_rule(self, X, Y, candidate_rule):
        covered_X = X[candidate_rule.covered_examples]
        possible_selectors = []

        for i, attribute in enumerate(candidate_rule.domain.attributes):
            column = covered_X[:, i]
            if attribute.is_discrete:
                for val in [int(x) for x in set(column)]:
                    s1 = Selector(column=i, op="==", value=val)
                    s2 = Selector(column=i, op="!=", value=val)
                    possible_selectors.extend([s1, s2])
            elif attribute.is_continuous:  # TODO: component based
                column = np.unique(column)
                dividers = np.array_split(column, min(10, column.shape[0]))
                dividers = [np.median(smh) for smh in dividers]
                for val in dividers:
                    s1 = Selector(column=i, op="<=", value=val)
                    s2 = Selector(column=i, op=">=", value=val)
                    possible_selectors.extend([s1, s2])

        possible_selectors = [smh for smh in possible_selectors if
                              smh not in candidate_rule.selectors]

        new_rules = []
        (target_class, selectors, domain, prior_class_dist,
         quality_evaluator, complexity_evaluator, significance_validator,
         general_validator) = candidate_rule.forward()

        for curr_selector in possible_selectors:
            copied_selectors = copy(selectors)
            copied_selectors.append(curr_selector)

            new_rule = Rule(selectors=copied_selectors,
                            parent_rule=candidate_rule, domain=domain,
                            prior_class_dist=prior_class_dist,
                            quality_evaluator=quality_evaluator,
                            complexity_evaluator=complexity_evaluator,
                            significance_validator=significance_validator,
                            general_validator=general_validator)

            new_rule.filter_and_store(X, Y, target_class, td_optimisation=True)
            # to ensure the covered_examples matrices are of the same size
            # throughout the RF iter.

            if new_rule.validity:
                new_rules.append(new_rule)

        return new_rules


class Selector:

    operators = {
        # discrete, nominal variables
        '==': operator.eq,
        '!=': operator.ne,

        # continuous variables
        '<=': operator.le,
        '>=': operator.ge
    }

    def __init__(self, column, op, value):
        self.column = column
        self.op = op
        self.value = value

    def filter_data(self, X):
        return self.operators[self.op](X[:, self.column], self.value)

    def filter_instance(self, x):
        return self.operators[self.op](x[self.column], self.value)

    def __eq__(self, other):
        return all((self.op == other.op, self.value == other.value,
                    self.column == other.column))


class Rule:

    """
    Represents a single rule and keeps a pointer to its parent. Taking
    into account numpy slicing and memory management, instance
    references are strictly not kept.

    Those can be easily gathered, however, by following the trail of
    covered examples from rule to rule if the original learning data
    reference is still known.
    """

    def __init__(self, selectors=None, parent_rule=None, domain=None,
                 prior_class_dist=None, quality_evaluator=None,
                 complexity_evaluator=None, significance_validator=None,
                 general_validator=None):

        """
        Initialises a Rule object.

        :param selectors:
        :param parent_rule:
        :param domain:
        :param prior_class_dist:
        :param quality_evaluator:
        :param complexity_evaluator:
        :param significance_validator:
        :param general_validator:
        """

        self.selectors = selectors if selectors is not None else []
        self.parent_rule = parent_rule
        self.domain = domain
        self.prior_class_dist = prior_class_dist
        self.quality_evaluator = quality_evaluator
        self.complexity_evaluator = complexity_evaluator
        self.significance_validator = significance_validator
        self.general_validator = general_validator

        self.target_class = None
        self.covered_examples = None
        self.curr_class_dist = None
        self.num_covered = None
        self.prediction = None
        self.quality = None
        self.complexity = None
        self.significance = None
        self.validity = None

    def filter_and_store(self, X, Y, target_class, td_optimisation=False):
        """
        Most rules will only ever make 1 call to this method before
        being discarded. Only rules returned by the rule_finder will
        ever have accessed any of the other class methods.
        """
        self.target_class = target_class
        if td_optimisation and self.parent_rule is not None:
            self.covered_examples = np.copy(self.parent_rule.covered_examples)
            start = len(self.parent_rule.selectors)
        else:
            self.covered_examples = np.ones(X.shape[0], dtype=np.bool)
            start = 0
        for selector in self.selectors[start:]:
            self.covered_examples &= selector.filter_data(X)

        self.curr_class_dist = get_dist(Y[self.covered_examples], self.domain)
        self.num_covered = (self.curr_class_dist[self.target_class]
                            if self.target_class is not None
                            else np.sum(self.curr_class_dist))

        self.validity = self.general_validator.validate_rule(self)
        if self.validity:
            self.quality = self.quality_evaluator.evaluate_rule(self)
            self.complexity = self.complexity_evaluator.evaluate_rule(self)
            self.significance = self.significance_validator.validate_rule(self)

    def evaluate_instance(self, x):
        # return True if the given instance matches the rule condition
        return all((selector.filter_instance(x) for selector in self.selectors))

    def evaluate_data(self, X):
        curr_covered = np.ones(X.shape[0], dtype=np.bool)
        for selector in self.selectors:
            curr_covered &= selector.filter_data(X)
        return curr_covered

    def create_model(self):
        self.prediction = (self.target_class if self.target_class is not None
                           else argmaxrnd(self.curr_class_dist))

    def forward(self):
        return (self.target_class, self.selectors, self.domain,
                self.prior_class_dist, self.quality_evaluator,
                self.complexity_evaluator, self.significance_validator,
                self.general_validator)

    def __str__(self):
        attributes = self.domain.attributes
        class_var = self.domain.class_var

        if self.selectors:
            cond = " AND ".join([attributes[s.column].name + s.op +
                                 (str(attributes[s.column].values[s.value])
                                  if attributes[s.column].is_discrete
                                  else str(s.value)) for s in self.selectors])
        else:
            cond = "TRUE"

        outcome = class_var.name + "=" + class_var.values[self.prediction]
        return "IF {} THEN {} ".format(cond, outcome)


class RuleFinder:

    """
    Learns a single rule from learning instances. Different rule
    learning algorithms can be characterised by changing the default
    components.
    """

    def __init__(self):
        self.search_algorithm = BeamSearchAlgorithm()
        self.search_strategy = TopDownSearch()

        # search heuristics
        self.quality_evaluator = EntropyEvaluator()
        self.complexity_evaluator = LengthEvaluator()
        # heuristics to avoid the over-fitting of noisy data
        self.significance_validator = LRSValidator()
        self.general_validator = CustomGeneralValidator()

    def __call__(self, X, Y, target_class, base_rules, domain):

        """
        Returns a single rule.

        :param X: learning instances
        :type X: np.ndarray
        :param Y: learning instances classification
        :type Y: np.ndarray
        :param base_rules: optional list of initial rules
        :type base_rules: a list of Rule objects
        :param target_class: index of a specific class to learn
        :param target_class: int
        :param domain: used to refine rules and to calculate class dists
        :type domain: Orange.data.domain.Domain
        :return: Rule object
        """
        def rcmp(rule): return rule.quality, rule.complexity
        prior_class_dist = get_dist(Y, domain)

        rules = self.search_strategy.initialize_rule(
            X, Y, target_class, base_rules, domain, prior_class_dist,
            self.quality_evaluator, self.complexity_evaluator,
            self.significance_validator, self.general_validator)

        rules = sorted(rules, key=rcmp, reverse=True)
        best_rule = rules[0]

        while len(rules) > 0:
            cand_rules, rules = self.search_algorithm.select_candidates(rules)
            for cand_rule in cand_rules:
                new_rules = self.search_strategy.refine_rule(X, Y, cand_rule)
                rules.extend(new_rules)
                for new_rule in new_rules:
                    if (new_rule.quality > best_rule.quality and
                            new_rule.significance):
                        best_rule = new_rule

            rules = sorted(rules, key=rcmp, reverse=True)
            rules = self.search_algorithm.filter_rules(rules)

        best_rule.create_model()
        return best_rule


class RuleLearner(Learner):

    """
    A base rule induction learner. Returns a rule classifier.

      - Fuernkranz J.; Separate-and-Conquer Rule Learning,
        Artificial Intelligence Review 13, 3-54, 1999

    Separate and conquer strategy is implemented, allowing for different
    rule learning algorithms to be easily defined by connecting together
    predefined components. In essence, learning instances are covered
    and removed following a chosen rule. The process is repeated while
    learning set examples remain. To evaluate found hypotheses and to
    choose the best rule in each iteration, search heuristics are used.
    Primarily, rule class distribution is the decisive determinant.

    The over-fitting of noisy data is avoided by preferring simpler,
    shorter rules even if the accuracy of more complex rules is higher.
    """

    __metaclass__ = ABCMeta

    def __init__(self, preprocessors=None, base_rules=None):

        """
        Initialises a RuleLearner object.

        Constrains the algorithm with a list of base rules. Also creates
        a RuleFinder object to set search bias and over-fitting
        avoidance bias parameters.

        :param base_rules: optional list of initial rules
        :type base_rules: a list of Rule objects
        """

        super().__init__(preprocessors=preprocessors)
        self.base_rules = base_rules if base_rules is not None else []
        self.rule_finder = RuleFinder()

    @abstractmethod
    def fit(self, X, Y, W=None):
        """ Descendants should override this method. """
        rule_list = self.find_rules(X, Y, None, self.base_rules, self.domain)
        return RuleClassifier(domain=self.domain, rule_list=rule_list)

    def find_rules(self, X, Y, target_class, base_rules, domain):
        rule_list = []
        while not self.data_stopping(X, Y, target_class):
            new_rule = self.rule_finder(X, Y, target_class, base_rules, domain)
            if self.rule_stopping(X, Y, new_rule):
                break
            X, Y = self.cover_and_remove(X, Y, new_rule)
            rule_list.append(new_rule)
        return rule_list

    @staticmethod
    def data_stopping(X, Y, target_class):
        """ Stop if no positive examples. """
        return Y.size == 0 or (target_class is not None and
                               target_class not in Y)

    @staticmethod
    def rule_stopping(X, Y, new_rule, alpha=1.0):
        return False

    @staticmethod
    def cover_and_remove(X, Y, new_rule):
        examples_to_keep = np.logical_not(new_rule.covered_examples)
        return X[examples_to_keep], Y[examples_to_keep]


class RuleClassifier(Model):

    """
    A rule induction classifier. Instances are classified following
    either an unordered set of rules or a decision list.
    """

    __metaclass__ = ABCMeta

    def __init__(self, domain=None, rule_list=None):
        super().__init__(domain)
        self.domain = domain
        self.rule_list = rule_list if rule_list is not None else []

    @abstractmethod
    def predict(self, X):
        """ Descendants should override this method. """
        classifications = []
        status = np.ones(X.shape[0], dtype=np.bool)
        for rule in self.rule_list:
            curr_covered = rule.evaluate_data(X)
            curr_covered &= status
            status &= np.bitwise_not(curr_covered)
            curr_covered[curr_covered] = rule.prediction
            classifications.append(curr_covered)
        return np.sum(np.row_stack(classifications), axis=0)


class CN2Learner(RuleLearner):

    """
    Classical CN2 inducer (Clark and Niblett; 1988) that constructs a
    set of ordered rules. Calling method fit with data returns a
    CN2Classifier.
    """

    def __init__(self, preprocessors=None, base_rules=None):
        super().__init__(preprocessors, base_rules)
        self.rule_finder.search_algorithm.beam_width = 10

    def fit(self, X, Y, W=None):
        rule_list = self.find_rules(X, Y, None, self.base_rules, self.domain)
        return CN2Classifier(domain=self.domain, rule_list=rule_list)


class CN2Classifier(RuleClassifier):
    def predict(self, X):
        return super().predict(X)


class CN2UnorderedLearner(RuleLearner):
    def __init__(self, preprocessors=None, base_rules=None):
        super().__init__(preprocessors, base_rules)
        self.rule_finder.search_algorithm.beam_width = 10
        self.rule_finder.quality_evaluator = LaplaceAccuracyEvaluator()

    def fit(self, X, Y, W=None):
        rule_list = []
        for curr_class in range(len(self.domain.class_var.values)):
            r = self.find_rules(X, Y, curr_class, self.base_rules, self.domain)
            rule_list.extend(r)
        return CN2UnorderedClassifier(domain=self.domain, rule_list=rule_list)

    @staticmethod
    def cover_and_remove(X, Y, new_rule):
        examples_to_keep = Y == new_rule.target_class
        examples_to_keep &= new_rule.covered_examples
        examples_to_keep = np.logical_not(examples_to_keep)
        return X[examples_to_keep], Y[examples_to_keep]


class CN2UnorderedClassifier(RuleClassifier):
    def predict(self, X):
        num_classes = len(self.domain.class_var.values)
        classifications = []
        status = np.ones(X.shape[0], dtype=np.bool)
        resolve_clash = np.zeros(X.shape[0], dtype=np.int)
        clashes = np.zeros((X.shape[0], num_classes), dtype=np.int)

        for rule in self.rule_list:
            curr_covered = rule.evaluate_data(X)
            resolve_clash += curr_covered
            temp = np.zeros((X.shape[0], num_classes), dtype=np.int)
            temp[curr_covered] = rule.curr_class_dist
            clashes += temp
            curr_covered &= status
            status &= np.bitwise_not(curr_covered)
            curr_covered[curr_covered] = rule.prediction
            classifications.append(curr_covered)

        resolve_clash = resolve_clash > 1
        no_clash = np.logical_not(resolve_clash)
        classifications = np.sum(np.row_stack(classifications), axis=0)
        result = np.zeros(X.shape[0], dtype=np.int)
        result[no_clash] = classifications[no_clash]
        result[resolve_clash] = argmaxrnd(clashes[resolve_clash])
        return result


def main():
    data = Orange.data.Table('titanic')

    learner = CN2Learner()
    classifier = learner(data)
    for rule in classifier.rule_list:
        print(rule.curr_class_dist.tolist(), rule)

    print()

    learner = CN2UnorderedLearner()
    classifier = learner(data)
    for rule in classifier.rule_list:
        print(rule, rule.curr_class_dist.tolist())

if __name__ == "__main__":
    main()
