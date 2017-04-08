# Models for word alignment

class TranslationModel:
    "Models conditional distribution over trg words given a src word, i.e. t(f|e)."

    def __init__(self, src_corpus, trg_corpus):
        self._src_trg_counts = {}
        self._trg_given_src_probs = {}

    def get_conditional_prob(self, src_token, trg_token):
        "Return the conditional probability of trg_token given src_token, i.e. t(f|e)."
        if src_token not in self._trg_given_src_probs:
            return 1.0
        if trg_token not in self._trg_given_src_probs[src_token]:
            return 1.0
        return self._trg_given_src_probs[src_token][trg_token]

    def collect_statistics(self, src_tokens, trg_tokens, posterior_matrix):
        "Accumulate counts of translations from matrix: matrix[j][i] = p(a_j=i|e, f)"
        assert len(posterior_matrix) == len(trg_tokens)
        for posterior in posterior_matrix:
            assert len(posterior) == len(src_tokens)
        # Hint - You just need to count how often each src and trg token are aligned
        # but since we don't have labeled data you'll use the posterior_matrix[j][i]
        # as the 'fractional' count for src_tokens[i] and trg_tokens[k].
        assert False, "Collect statistics here!"

    def recompute_parameters(self):
        "Reestimate parameters and reset counters."
        # Hint - Just normalize the self._src_and_trg_counts so that the conditional
        # distributions self._trg_given_src_probs are correctly normalized to give t(f|e).
        assert False, "Recompute parameters here."

class PriorModel:
    "Models the prior probability of an alignment given only the sentence lengths and token indices."

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for more sophisticated models."
        self._distance_counts = {}
        self._distance_probs = {}

    def get_prior_prob(self, src_index, trg_index, src_length, trg_length):
        "Returns a uniform prior probability."
        # Hint - you can probably improve on this, but this works as is.
        return 1.0/ src_length

    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        "Count the necessary statistics from this matrix if needed."
        pass

    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        pass
