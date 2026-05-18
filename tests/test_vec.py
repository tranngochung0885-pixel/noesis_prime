from noesis_prime import Vec


def test_vec_normalize_zero_safe():
    v = [0.0, 0.0, 0.0]
    out = Vec.normalize(v)
    assert len(out) == 3


def test_vec_cosine_identity():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(Vec.cosine(a, b) - 1.0) < 1e-6