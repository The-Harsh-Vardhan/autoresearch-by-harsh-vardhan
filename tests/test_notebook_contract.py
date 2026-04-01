from autoresearch_hv.hndsr_vr.notebook_contract import validate_versioned_notebook


def test_vr1_notebook_contract_passes():
    failures = validate_versioned_notebook(
        notebook_path="notebooks/versions/vR.1_HNDSR.ipynb",
        doc_path="docs/notebooks/vR.1_HNDSR.md",
        review_path="reports/reviews/vR.1_HNDSR.review.md",
        full_config_path="configs/hndsr_vr/vR.1_train.yaml",
        smoke_config_path="configs/hndsr_vr/vR.1_smoke.yaml",
        control_config_path="configs/hndsr_vr/vR.1_control.yaml",
        version="vR.1",
    )
    assert failures == []
