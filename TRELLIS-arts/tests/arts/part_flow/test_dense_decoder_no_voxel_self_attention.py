from trellis.models.part_flow.part_flow_predictor import (
    DenseVoxelDecoderLayer,
    PartFlowPredictor,
    PartTokenTransformer,
)


def test_dense_decoder_layer_has_no_voxel_self_attention_weights():
    layer = DenseVoxelDecoderLayer(dim=32, num_heads=4, dropout=0.0)
    names = dict(layer.named_parameters()).keys()
    forbidden = ('self_q', 'self_k', 'self_v', 'self_o')
    assert not any(any(f in name for f in forbidden) for name in names)


def test_predictor_uses_part_token_transformer_and_four_dense_layers():
    model = PartFlowPredictor(
        k_max=8,
        hidden_dim=32,
        num_layers=4,
        num_heads=4,
        cond_dim=16,
        num_views=4,
    )
    assert isinstance(model.part_token_transformer, PartTokenTransformer)
    assert len(model.decoder_layers) == 4
    assert all(isinstance(layer, DenseVoxelDecoderLayer) for layer in model.decoder_layers)
