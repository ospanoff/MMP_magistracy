#!/usr/bin/python
#
# The methods in this file (once implemented) will perform the EM optimization
# to compute the parameters of a word alignment model and to output word alignments.
#
# A note on notation used in comments.
#   e = src_tokens, i = src_index (so src_tokens[src_index] = e_i)
#   f = trg_tokens, j = trg_index (so trg_tokens[trg_index] = f_j)
#   a_j = alignment index of f_j (i.e. an index into src_tokens)

import os, sys, codecs, utils
from math import log
from models import PriorModel
from models import TranslationModel

def get_posterior_distribution_for_trg_token(trg_index, src_tokens, trg_tokens,
                                             prior_model, translation_model):
    "Compute the posterior distribution over alignments for trg_index: P(a_j = i|f_j, e)."
    assert False, "Implement this."
    # Compute the following two values given the current model parameters (E-step)
    #   marginal_prob = p(f_j|e) = sum over i p(f_j, a_j = i|e)
    #   posterior_probs[i] = p(a_j = i|f_j, e) = p(f_j, a_j = i|e) / p(f_j|e)
    #
    # Hint:
    #  joint_probs[i] = p(f_j, a_j = i|e) = p(a_j = i|e) p(f_j|a_j=i, e)
    #  - use the prior_model.get_prior_prob method to compute p(a_j = i|e)
    #  - use the translation_model.get_conditional_prob method to compute p(f_j|a_j = i, e) 
    return marginal_prob, posterior_probs

def get_posterior_alignment_matrix(src_tokens, trg_tokens, prior_model, translation_model):
    "For each target token compute the posterior alignment probability: p(a_j=i | f_j, e)"
    # Since these models assume each trg token is aligned independently, this method just
    # calls get_posterior_distribution_for_trg_token for each trg token.
    posterior_matrix = [] # posterior_matrix[j][i] = p(a_j=i | f_j, e).
    sentence_marginal_log_likelihood = 0.0
    for trg_index, trg_token in enumerate(trg_tokens):
        # Compute posterior probability that this trg token is alignment to each src token.
        marginal_prob, posterior_distribution = get_posterior_distribution_for_trg_token(
            trg_index, src_tokens, trg_tokens, prior_model, translation_model)
        posterior_matrix.append(posterior_distribution)
        # This keeps track of the log likelihood.
        sentence_marginal_log_likelihood += log(marginal_prob)
    return sentence_marginal_log_likelihood, posterior_matrix

def collect_expected_statistics(src_corpus, trg_corpus, prior_model, translation_model):
    "Infer posterior distribution over each sentence pair and collect statistics: E-step"
    corpus_marginal_log_likelihood = 0.0
    for i in range(len(src_corpus)):
        # Infer posterior
        sentence_marginal_log_likelihood, posterior_matrix = get_posterior_alignment_matrix(
            src_corpus[i], trg_corpus[i], prior_model, translation_model)
        # Collect statistics in each model.
        prior_model.collect_statistics(len(src_corpus[i]), len(trg_corpus[i]), posterior_matrix)
        translation_model.collect_statistics(src_corpus[i], trg_corpus[i], posterior_matrix)
        # Update log prob
        corpus_marginal_log_likelihood += sentence_marginal_log_likelihood
    return corpus_marginal_log_likelihood

def reestimate_models(prior_model, translation_model):
    "Recompute parameters of each model: M-step"
    prior_model.recompute_parameters()
    translation_model.recompute_parameters()

def initialize_models(src_corpus, trg_corpus):
    prior_model = PriorModel(src_corpus, trg_corpus)
    translation_model = TranslationModel(src_corpus, trg_corpus)
    return prior_model, translation_model

def estimate_models(src_corpus, trg_corpus, prior_model, translation_model, num_iterations):
    "Estimate models iteratively."
    for iteration in range(num_iterations):
        corpus_log_likelihood = collect_expected_statistics(
            src_corpus, trg_corpus, prior_model, translation_model)
        reestimate_models(prior_model, translation_model)
        if iteration > 0:
            print "corpus log likelihood: %1.3f" % corpus_log_likelihood
    return prior_model, translation_model

def align_sentence_pair(src_tokens, trg_tokens, prior_probs, translation_probs):
    "For each target token, find the src_token with the highest posterior probability."
    # Compute the posterior distribution over alignments for all target tokens.
    corpus_log_prob, posterior_matrix = get_posterior_alignment_matrix(
        src_tokens, trg_tokens, prior_probs, translation_probs)
    # For each target word find the src index with the highest posterior probability.
    alignments = []
    for trg_index, posteriors in enumerate(posterior_matrix):
        best_src_index = posteriors.index(max(posteriors))
        alignments.append((best_src_index, trg_index))
    return alignments

def align_corpus_given_models(src_corpus, trg_corpus, prior_model, translation_model):
    "Align each sentence pair in the corpus in turn."
    alignments = []
    for i in range(len(src_corpus)):
        these_alignments = align_sentence_pair(
            src_corpus[i], trg_corpus[i], prior_model, translation_model)
        alignments.append(these_alignments)
    return alignments

def align_corpus(src_corpus, trg_corpus, num_iterations):
    "Learn models and then align the corpus using them."
    prior_model, translation_model = initialize_models(src_corpus, trg_corpus)
    prior_model, translation_model = estimate_models(
        src_corpus, trg_corpus, prior_model, translation_model, num_iterations)
    return align_corpus_given_models(src_corpus, trg_corpus, prior_model, translation_model)


if __name__ == "__main__":
    if not len(sys.argv) == 5:
        print "Usage ./word_alignment.py src_corpus trg_corpus iterations output_prefix."
        sys.exit(0)
    src_corpus = utils.read_all_tokens(sys.argv[1])
    trg_corpus = utils.read_all_tokens(sys.argv[2])
    num_iterations = int(sys.argv[3])
    output_prefix = sys.argv[4]
    assert len(src_corpus) == len(trg_corpus), "Corpora should be same size!"
    alignments = align_corpus(src_corpus, trg_corpus, num_iterations)
    utils.output_alignments_per_test_set(alignments, output_prefix)

