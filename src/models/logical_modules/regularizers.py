"""Logical Module Regularizers"""

# ============================================================
#                      NOT REGULARIZERS
# ============================================================


def r1_negation(NOT, X, sim):
    """r1 = 1/|X| ∑ (1 + Sim(NOT(x), x))"""
    not_x = NOT(X)
    return (1.0 + sim(not_x, X)).mean()


def r2_double_negation(NOT, X, sim):
    """r2 = 1/|X| ∑ (1 - Sim(NOT(NOT(x)), x))"""
    not_not_x = NOT(NOT(X))
    return (1.0 - sim(not_not_x, X)).mean()


def not_regularizer(NOT, X, sim):
    """Combined regularizer for NOT module."""
    return r1_negation(NOT, X, sim) + r2_double_negation(NOT, X, sim)


# ============================================================
#                      AND REGULARIZERS
# ============================================================


def r3_and_identity(AND, TRUE, X, sim):
    """r3 = 1/|X| ∑ (1 - Sim(AND(x, TRUE), x))"""
    T = TRUE.unsqueeze(0).expand(X.size(0), -1)
    and_x_t = AND(X, T)
    return (1.0 - sim(and_x_t, X)).mean()


def r4_and_annihilator(AND, FALSE, X, sim):
    """r4 = 1/|X| ∑ (1 - Sim(AND(x, FALSE), FALSE))"""
    F_exp = FALSE.unsqueeze(0).expand(X.size(0), -1)
    and_x_f = AND(X, F_exp)
    return (1.0 - sim(and_x_f, F_exp)).mean()


def r5_and_idempotence(AND, X, sim):
    """r5 = 1/|X| ∑ (1 - Sim(AND(x, x), x))"""
    and_x_x = AND(X, X)
    return (1.0 - sim(and_x_x, X)).mean()


def r6_and_complement(AND, NOT, TRUE, X, sim):
    """r6 = 1/|X| ∑ (1 - Sim(AND(x, NOT(x)), FALSE))"""
    FALSE = NOT(TRUE.unsqueeze(0)).squeeze(0)
    not_X = NOT(X)
    and_x_notx = AND(X, not_X)
    F_exp = FALSE.unsqueeze(0).expand(X.size(0), -1)
    return (1.0 - sim(and_x_notx, F_exp)).mean()


def and_regularizer(AND, NOT, TRUE, X, sim):
    """Combined regularizer for AND module."""
    FALSE = NOT(TRUE.unsqueeze(0)).squeeze(0)
    return (
        r3_and_identity(AND, TRUE, X, sim)
        + r4_and_annihilator(AND, FALSE, X, sim)
        + r5_and_idempotence(AND, X, sim)
        + r6_and_complement(AND, NOT, TRUE, X, sim)
    )


# ============================================================
#                      OR REGULARIZERS
# ============================================================


def r7_or_identity(OR, FALSE, X, sim):
    """r7 = 1/|X| ∑ (1 - Sim(OR(x, FALSE), x))"""
    F_exp = FALSE.unsqueeze(0).expand(X.size(0), -1)
    or_x_f = OR(X, F_exp)
    return (1.0 - sim(or_x_f, X)).mean()


def r8_or_annihilator(OR, TRUE, X, sim):
    """r8 = 1/|X| ∑ (1 - Sim(OR(x, TRUE), TRUE))"""
    T_exp = TRUE.unsqueeze(0).expand(X.size(0), -1)
    or_x_t = OR(X, T_exp)
    return (1.0 - sim(or_x_t, T_exp)).mean()


def r9_or_idempotence(OR, X, sim):
    """r9 = 1/|X| ∑ (1 - Sim(OR(x, x), x))"""
    or_x_x = OR(X, X)
    return (1.0 - sim(or_x_x, X)).mean()


def r10_or_complement(OR, NOT, TRUE, X, sim):
    """r10 = 1/|X| ∑ (1 - Sim(OR(x, NOT(x)), TRUE))"""
    not_X = NOT(X)
    T_exp = TRUE.unsqueeze(0).expand(X.size(0), -1)
    or_x_notx = OR(X, not_X)
    return (1.0 - sim(or_x_notx, T_exp)).mean()


def or_regularizer(OR, NOT, TRUE, X, sim):
    """Combined regularizer for OR module."""
    FALSE = NOT(TRUE.unsqueeze(0)).squeeze(0)
    return (
        r7_or_identity(OR, FALSE, X, sim)
        + r8_or_annihilator(OR, TRUE, X, sim)
        + r9_or_idempotence(OR, X, sim)
        + r10_or_complement(OR, NOT, TRUE, X, sim)
    )
