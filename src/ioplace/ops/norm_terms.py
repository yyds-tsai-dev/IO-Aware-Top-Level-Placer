"""Adapters exposing the production IO and FT terms to `norm.TermNormalizer`.

The normalizer's term protocol is a single method, `value(pos, ctx) -> Tensor`,
returning the term's *unweighted* objective: the normalizer owns the backward,
the fixed/filler masking and the norm order, so every term's gradient norm is
measured identically (design sec 4).
"""


class IoNormTerm(object):
    """Unweighted `L_IO` (`lambda_io=1`, no margin) at the schedule's live tau."""

    name = "io"

    def __init__(self, io_term):
        self.io_term = io_term

    def value(self, pos, ctx):
        return self.io_term(pos, ctx["tau"], 1.0)


class FtNormTerm(object):
    """Unweighted feed-through part only -- `FtTerm.ft_only`, i.e. the same
    isolated quantity `ops/ft_callback.publish_atomic` measures for `kappa_ft`."""

    name = "ft"

    def __init__(self, ft_term):
        self.ft_term = ft_term

    def value(self, pos, ctx):
        return self.ft_term.ft_only(pos, ctx["tau"])
