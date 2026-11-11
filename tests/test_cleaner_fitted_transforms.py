import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from MAna.data import DataCleaner


class FittedTransformTests(unittest.TestCase):
    def make_cleaner(self):
        return DataCleaner(
            pd.DataFrame({'amount': [10., 20., 40.], 'region': ['b', 'a', 'b']},
                         index=[3, 5, 8]),
            numerical_columns=['amount'], categorical_columns=['region'],
            verbose=False,
        )

    def test_roundtrip_and_new_data_without_refit(self):
        for method in ['onehot', 'label', 'ordinal']:
            with self.subTest(method=method):
                cleaner = self.make_cleaner()
                original = cleaner.df.copy()
                cleaner.encode_categorical(method=method).scale_features(method='robust')
                assert_frame_equal(cleaner.inverse_transform(), original)
                new = pd.DataFrame({'amount': [100.], 'region': ['a']}, index=[42])
                center = cleaner.get_scaler('amount').center_.copy()
                encoded = cleaner.transform(new)
                assert_frame_equal(cleaner.inverse_transform(encoded), new)
                self.assertEqual(cleaner.get_scaler('amount').center_[0], center[0])
                self.assertIsNotNone(cleaner.get_encoder('region'))

    def test_drop_first_and_scaling_encoded_columns(self):
        cleaner = self.make_cleaner()
        original = cleaner.df.copy()
        cleaner.encode_categorical(drop_first=True)
        cleaner.scale_features(columns=list(cleaner.df.columns))
        assert_frame_equal(cleaner.inverse_transform(), original)

    def test_separate_label_encoders(self):
        frame = pd.DataFrame({'a': ['x', 'y'], 'b': ['m', 'n']})
        cleaner = DataCleaner(frame, categorical_columns=['a', 'b'], verbose=False)
        cleaner.encode_categorical(method='label')
        self.assertIsNot(cleaner.get_encoder('a'), cleaner.get_encoder('b'))
        assert_frame_equal(cleaner.inverse_transform(), frame)

    def test_persistence_and_reset(self):
        cleaner = self.make_cleaner().encode_categorical().scale_features()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'fitted.joblib')
            cleaner.save_fitted_pipeline(path)
            loaded = DataCleaner.load_fitted_pipeline(path)
            assert_frame_equal(loaded.inverse_transform(), cleaner.df_original)
            assert_frame_equal(loaded.transform(cleaner.df_original), cleaner.df)
        cleaner.reset()
        self.assertEqual(cleaner.scalers, {})
        self.assertEqual(cleaner.encoders, {})

    def test_unsupported_cleaning_and_unknown_categories_raise(self):
        cleaner = self.make_cleaner().encode_categorical()
        with self.assertRaises(ValueError):
            cleaner.transform(pd.DataFrame({'amount': [2.], 'region': ['unknown']}))
        cleaner.remove_duplicates()
        with self.assertRaisesRegex(ValueError, 'Cannot replay'):
            cleaner.transform(cleaner.df_original)

    def test_target_encoding_inverse_is_explicitly_unsupported(self):
        cleaner = DataCleaner(
            pd.DataFrame({'region': ['a', 'b'], 'target': [0, 1]}),
            target_column='target', categorical_columns=['region'], verbose=False,
        ).encode_categorical(method='target')
        with self.assertRaisesRegex(ValueError, 'not uniquely invertible'):
            cleaner.inverse_transform()
