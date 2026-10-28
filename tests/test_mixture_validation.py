import unittest
import tempfile
from pathlib import Path
import pandas as pd
from mixture_validation import validate_models, validate_predictions, SETS
from analyze_mixture_results import build_review

class MixtureValidationTests(unittest.TestCase):
    def test_missing_seed_rejected_before_artifacts(self):
        with self.assertRaisesRegex(ValueError, 'exactly'):
            validate_models(pd.DataFrame([dict(feature_set=s,seed=11) for s in SETS]))

    def test_duplicate_model_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_models(pd.DataFrame([dict(feature_set='set_c_core_shape',seed=11)]*2))

    def test_changed_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.joblib'
            path.write_bytes(b'changed artifact')
            models = pd.DataFrame([dict(feature_set=s,seed=n,model_path=str(path),model_sha256='wrong',
                                        n_hsa=432,n_f8=432,n_total=864) for s in SETS for n in [11,42]])
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                validate_models(models)

    def test_missing_prediction_rejected(self):
        rows=[dict(event_id='a',feature_set=s,seed=n,probability_f8=.2) for s in SETS for n in [11,42]]
        validate_predictions(pd.DataFrame(rows),['a'])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            validate_predictions(pd.DataFrame(rows[:-1]),['a'])

    def test_review_includes_seed_and_non_c_uncertainty(self):
        e=pd.DataFrame([dict(event_id='a',mixture_id='m',recording_id='r',feature_set=s,
                            probability_f8_mean=p,seed_agreement=agree,predicted_class='HSA')
                        for s,p,agree in [('set_c_core_shape',.1,True),('set_d_core_segments',.45,True),('set_e_segment_only',.2,False)]])
        r=build_review(e,.5).iloc[0]
        self.assertTrue(r.near_threshold)
        self.assertTrue(r.seed_disagreement)
        self.assertFalse(r.feature_set_disagreement)

if __name__=='__main__': unittest.main()
