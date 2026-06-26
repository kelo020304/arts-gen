from trellis.trainers.arts.part_flow import _mean_eval_scalars


def test_mean_eval_scalars_uses_unweighted_sample_mean():
    rows = [
        {
            'acc': 0.2,
            'non_empty_acc': 0.4,
            'target_acc': 0.6,
            'body_acc': 0.8,
            'mIoU': 0.1,
            'non_empty_mIoU': 0.3,
            'target_mIoU': 0.5,
        },
        {
            'acc': 0.4,
            'non_empty_acc': 0.6,
            'target_acc': 0.8,
            'body_acc': 1.0,
            'mIoU': 0.3,
            'non_empty_mIoU': 0.5,
            'target_mIoU': 0.7,
        },
    ]

    got = _mean_eval_scalars(rows)

    assert abs(got['acc'] - 0.3) < 1e-8
    assert abs(got['non_empty_acc'] - 0.5) < 1e-8
    assert abs(got['target_acc'] - 0.7) < 1e-8
    assert abs(got['body_acc'] - 0.9) < 1e-8
    assert abs(got['mIoU'] - 0.2) < 1e-8
    assert abs(got['non_empty_mIoU'] - 0.4) < 1e-8
    assert abs(got['target_mIoU'] - 0.6) < 1e-8
