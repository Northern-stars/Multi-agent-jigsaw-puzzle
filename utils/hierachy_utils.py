def has_any_gradient(model):
    return any(param.grad is not None for param in model.parameters() if param.requires_grad)


def selective_load_state_dict(source_model, target_model, layer_mapping):
    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    new_state_dict = target_state_dict.copy()

    for src_layer, tgt_layer in layer_mapping.items():
        src_weight_key = f"{src_layer}.weight"
        src_bias_key = f"{src_layer}.bias"
        tgt_weight_key = f"{tgt_layer}.weight"
        tgt_bias_key = f"{tgt_layer}.bias"

        if src_weight_key in source_state_dict and tgt_weight_key in new_state_dict:
            if source_state_dict[src_weight_key].shape == new_state_dict[tgt_weight_key].shape:
                new_state_dict[tgt_weight_key] = source_state_dict[src_weight_key].clone()
            else:
                print(f"权重形状不匹配: {src_weight_key} -> {tgt_weight_key}")

        if src_bias_key in source_state_dict and tgt_bias_key in new_state_dict:
            if source_state_dict[src_bias_key].shape == new_state_dict[tgt_bias_key].shape:
                new_state_dict[tgt_bias_key] = source_state_dict[src_bias_key].clone()
            else:
                print(f"偏置形状不匹配: {src_bias_key} -> {tgt_bias_key}")

    target_model.load_state_dict(new_state_dict, strict=False)
    return target_model
