from noesis_prime import NOESISConfig


def test_config_defaults():
    cfg = NOESISConfig(embed_dim=128, wm_dims=(128, 64, 32, 16), self_model_dim=16)
    assert cfg.embed_dim == 128
    assert len(cfg.wm_dims) == 4
    assert cfg.self_model_dim == 16