from fabrique.loading import ConversionRule, IGNORE, maybe_apply_rule


def check_pattern(safe_pattern: str, safe_key: str, should_match: bool):
    rule = ConversionRule(safe_pattern, IGNORE)
    path_and_val = maybe_apply_rule(rule, safe_key, 42)
    if should_match:
        assert isinstance(path_and_val, tuple), f"Key {safe_key} didn't match pattern {safe_pattern}"
    else:
        assert path_and_val is None, f"Key {safe_key} wrongly matched pattern {safe_pattern}"



def test_safe_regexp():
    check_pattern("encoder.dense.weight", "encoder.dense.weight", True)
    check_pattern("encoder.dense.weight", "encoder.dense.bias", False)
    check_pattern("layers.{n}.dense.weight", "layers.0.dense.weight", True)
    check_pattern("layers.{n}.dense.weight", "layers.0.dense.bias", False)
    check_pattern("lm_head.*", "lm_head.bias", True)
    check_pattern("lm_head.*", "lm_head.dense.bias", True)
    check_pattern("lm_head.*", "cls.dense.bias", False)


