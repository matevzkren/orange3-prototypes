from copy import copy
import operator
import numpy as np
from scipy.stats import chisqprob
from orangecontrib.prototypes.rule_learning._argmaxrnd import argmaxrnd

import Orange
from Orange.classification import Learner, Model


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


def get_distribution(Y, domain):
    return np.bincount(Y.astype(dtype=np.int), minlength=len(domain.class_var.values))


def rule_length(rule):
    return len(rule.selectors)


class Evaluator:
    def evaluate_rule(self, rule):
        raise NotImplementedError


class EntropyEvaluator(Evaluator):
    def evaluate_rule(self, rule):
        x = rule.curr_class_distribution.astype(dtype=np.float)
        if rule.target_class is not None:
            x = np.array([x[rule.target_class], np.sum(x) - x[rule.target_class]], dtype=np.float)
        return -entropy(x)


class LengthEvaluator(Evaluator):
    def evaluate_rule(self, rule):
        return -rule_length(rule)


class Validator:
    def validate_rule(self, rule):
        raise NotImplementedError


class CustomGeneralValidator(Validator):
    """
    Discard rules that

    - cover less than the minimum required number of examples,
    - are too complex,
    - or offer no additional advantage in comparison to the parent rule.

    """

    def __init__(self, max_rule_length=5, minimum_covered_examples=1):
        self.max_rule_length = max_rule_length
        self.minimum_covered_examples = minimum_covered_examples

    def validate_rule(self, rule):
        nr_covered = np.sum(rule.curr_class_distribution)

        return (nr_covered >= self.minimum_covered_examples and
                rule_length(rule) <= self.max_rule_length and
                nr_covered != np.sum(rule.parent_rule.curr_class_distribution)
                if rule.parent_rule is not None else True)


class LRSValidator(Validator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def validate_rule(self, rule):
        if self.alpha >= 1.0 or rule.parent_rule is None:
            return True

        x = rule.curr_class_distribution.astype(dtype=np.float)
        y = rule.parent_rule.curr_class_distribution.astype(dtype=np.float)
        if rule.target_class is not None:
            x = np.array([x[rule.target_class], np.sum(x) - x[rule.target_class]], dtype=np.float)
            y = np.array([y[rule.target_class], np.sum(y) - y[rule.target_class]], dtype=np.float)

        lrs = likelihood_ratio_statistic(x, y)
        return lrs > 0 and chisqprob(lrs, len(rule.curr_class_distribution) - 1) <= self.alpha


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
    def initialize_rule(self, X, Y, target_class, base_rules, domain, prior_class_distribution,
                        quality_evaluator, complexity_evaluator, significance_validator,
                        general_validator):
        raise NotImplementedError

    def refine_rule(self, X, Y, candidate_rule):
        raise NotImplementedError


class TopDownSearch(SearchStrategy):
    def initialize_rule(self, X, Y, target_class, base_rules, domain, prior_class_distribution,
                        quality_evaluator, complexity_evaluator, significance_validator,
                        general_validator):

        rules = []

        general_rule = Rule(domain=domain, prior_class_distribution=prior_class_distribution,
                            quality_evaluator=quality_evaluator,
                            complexity_evaluator=complexity_evaluator,
                            significance_validator=significance_validator,
                            general_validator=general_validator)

        general_rule.filter_and_store(X, Y, target_class)
        rules.append(general_rule)

        for base_rule in base_rules:
            temp_rule = Rule(selectors=copy(base_rule.selectors), parent_rule=general_rule,
                             domain=domain, prior_class_distribution=prior_class_distribution,
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
        covered_Y = Y[candidate_rule.covered_examples]
        possible_domain_selectors = []

        for i, attribute in enumerate(candidate_rule.domain.attributes):
            temp_X = covered_X[:, i]

            if attribute.is_discrete:
                for value in [int(x) for x in set(temp_X)]:
                    possible_domain_selectors.append(Selector(column=i, op="==", value=value))
                    possible_domain_selectors.append(Selector(column=i, op="!=", value=value))

            elif attribute.is_continuous:
                # TODO: create ways (components) to discretise differently
                nr_intervals = np.min(10, len(temp_X))
                for value in [np.median(smh) for smh in np.split(np.sort(temp_X), nr_intervals)]:
                    possible_domain_selectors.append(Selector(column=i, op="<=", value=value))
                    possible_domain_selectors.append(Selector(column=i, op=">=", value=value))

        # TODO: check for conflicting selectors - measure time & decide
        possible_domain_selectors = [smh for smh in possible_domain_selectors if
                                     smh not in candidate_rule.selectors]

        new_rules = []
        target_class, selectors, domain, prior_class_distribution, quality_evaluator, \
            complexity_evaluator, significance_validator, general_validator = candidate_rule.forward()

        for curr_selector in possible_domain_selectors:
            copied_selectors = copy(selectors)
            copied_selectors.append(curr_selector)

            new_rule = Rule(selectors=copied_selectors, parent_rule=candidate_rule, domain=domain,
                            prior_class_distribution=prior_class_distribution,
                            quality_evaluator=quality_evaluator,
                            complexity_evaluator=complexity_evaluator,
                            significance_validator=significance_validator,
                            general_validator=general_validator)

            new_rule.filter_and_store(X, Y, target_class, td_optimisation=True)
            # to ensure the covered_examples matrices are of the same size throughout the RF iter.

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
    def __init__(self, selectors=None, parent_rule=None, domain=None, prior_class_distribution=None,
                 quality_evaluator=None, complexity_evaluator=None, significance_validator=None,
                 general_validator=None):

        self.selectors = selectors if selectors is not None else []
        self.parent_rule = parent_rule
        self.domain = domain
        self.prior_class_distribution = prior_class_distribution
        self.quality_evaluator = quality_evaluator
        self.complexity_evaluator = complexity_evaluator
        self.significance_validator = significance_validator
        self.general_validator = general_validator

        self.target_class = None
        self.covered_examples = None
        self.curr_class_distribution = None
        self.quality = None
        self.complexity = None
        self.significance = None
        self.validity = None

        self.prediction = None

    def filter_and_store(self, X, Y, target_class, td_optimisation=False):
        self.target_class = target_class

        if td_optimisation and self.parent_rule is not None:
            self.covered_examples = np.copy(self.parent_rule.covered_examples)
            start = len(self.parent_rule.selectors)
        else:
            self.covered_examples = np.ones(X.shape[0], dtype=np.bool)
            start = 0

        for selector in self.selectors[start:]:
            self.covered_examples &= selector.filter_data(X)

        self.curr_class_distribution = get_distribution(Y[self.covered_examples], self.domain)
        self.validity = self.general_validator.validate_rule(self)

        if self.validity:
            self.quality = self.quality_evaluator.evaluate_rule(self)
            self.complexity = self.complexity_evaluator.evaluate_rule(self)
            self.significance = self.significance_validator.validate_rule(self)

    def evaluate_instance(self, x):
        # return True if the given instance matches the rule condition
        return all((selector.filter_instance(x) for selector in self.selectors))

    def create_model(self):
        self.prediction = argmaxrnd(self.curr_class_distribution)

    def forward(self):
        return self.target_class, self.selectors, self.domain, self.prior_class_distribution,\
               self.quality_evaluator, self.complexity_evaluator, self.significance_validator, \
               self.general_validator

    def __str__(self):
        attributes = self.domain.attributes
        class_var = self.domain.class_var

        if self.selectors:
            conditions = " AND ".join([attributes[s.column].name + s.op +
                                       str(attributes[s.column].values[s.value])
                                       for s in self.selectors])
        else:
            conditions = "TRUE"

        outcome = class_var.name + "=" + class_var.values[self.prediction]
        return "IF {} THEN {} ".format(conditions, outcome)


class RuleFinder:
    def __init__(self):
        self.search_algorithm = BeamSearchAlgorithm()
        self.search_strategy = TopDownSearch()

        self.quality_evaluator = EntropyEvaluator()  # search heuristics 1
        self.complexity_evaluator = LengthEvaluator()
        self.significance_validator = LRSValidator()  # search heuristics 2
        self.general_validator = CustomGeneralValidator()

    def __call__(self, X, Y, target_class, base_rules, domain):
        prior_class_distribution = get_distribution(Y, domain)

        rules = self.search_strategy.initialize_rule(
            X, Y, target_class, base_rules, domain, prior_class_distribution, self.quality_evaluator,
            self.complexity_evaluator, self.significance_validator, self.general_validator)

        rules = sorted(rules, key=lambda x: (x.quality, x.complexity), reverse=True)
        best_rule = rules[0]

        while len(rules) > 0:
            candidate_rules, rules = self.search_algorithm.select_candidates(rules)
            for candidate_rule in candidate_rules:
                new_rules = self.search_strategy.refine_rule(X, Y, candidate_rule)
                rules.extend(new_rules)
                for new_rule in new_rules:
                    if new_rule.quality > best_rule.quality and new_rule.significance:
                        best_rule = new_rule

            rules = sorted(rules, key=lambda x: (x.quality, x.complexity), reverse=True)
            rules = self.search_algorithm.filter_rules(rules)

        # TODO: sever ties with the parent rule?
        best_rule.create_model()
        return best_rule


class RuleLearner(Learner):
    def __init__(self, preprocessors=None, base_rules=None, store_instances=False,
                 target_class=None):

        super().__init__(preprocessors=preprocessors)
        self.base_rules = base_rules if base_rules is not None else []
        self.store_instances = store_instances
        self.target_class = target_class
        self.rule_finder = RuleFinder()

    def fit(self, X, Y, W=None):
        rule_list = []

        while not self.data_stopping(X, Y, self.target_class):
            new_rule = self.rule_finder(X, Y, self.target_class, self.base_rules, self.domain)
            if self.rule_stopping(X, Y, new_rule):
                break
            X, Y = self.cover_and_remove(X, Y, new_rule)
            rule_list.append(new_rule)

        classifier = self.create_classifier(rule_list)
        return classifier

    def create_classifier(self, rule_list):
        return RuleClassifier(domain=self.domain, rule_list=rule_list)

    @staticmethod
    def data_stopping(X, Y, target_class):
        """ Stop if no positive examples. """
        return Y.size == 0 or target_class is not None and target_class not in Y

    @staticmethod
    def rule_stopping(X, Y, new_rule, alpha=1.0):
        return False

    @staticmethod
    def cover_and_remove(X, Y, new_rule):
        examples_to_keep = np.logical_not(new_rule.covered_examples)
        return X[examples_to_keep], Y[examples_to_keep]


class RuleClassifier(Model):
    def __init__(self, domain=None, rule_list=None):
        super().__init__(domain)
        self.rule_list = rule_list if rule_list is not None else []

    def predict(self, X):
        Y = []
        for x in X:
            for rule in self.rule_list:
                if rule.evaluate_instance(x):
                    Y.append(rule.prediction)
                    break
        return np.asarray(Y)


class CN2Learner(RuleLearner):
    def __init__(self, preprocessors=None, base_rules=None, store_instances=False,
                 target_class=None):
        super().__init__(preprocessors, base_rules, store_instances, target_class)
        self.rule_finder.search_algorithm.beam_width = 10

    def create_classifier(self, rule_list):
        return CN2Classifier(domain=self.domain, rule_list=rule_list)


class CN2Classifier(RuleClassifier):
    pass


def main():
    titanic = Orange.data.Table('titanic')

    # rf = RuleFinder()
    # rf.search_algorithm = BeamSearchAlgorithm(beam_width=10)
    #
    # rule_learner = RuleLearner()
    # rule_learner.rule_finder = rf
    # rule_classifier = rule_learner(titanic)
    #
    # for rule in rule_classifier.rule_list:
    #     print(rule.curr_class_distribution, rule)

    learner = CN2Learner()
    classifier = learner(titanic)
    for rule in classifier.rule_list:
        print(rule.curr_class_distribution.tolist(), rule)

if __name__ == "__main__":
    main()
